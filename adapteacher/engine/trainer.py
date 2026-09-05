# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.structures import ImageList
import numpy as np
from adapteacher.engine.save_pseudo import append_batch_with_filenames
from collections import OrderedDict
from fvcore.nn import FlopCountAnalysis
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from adapteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)

from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from adapteacher.engine.hooks import BestCheckpointer
from adapteacher.engine.hooks import LossEvalHook
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from adapteacher.solver.build import build_lr_scheduler
from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
from adapteacher.modeling.custom_losses import ConsistencyLosses

from .probe import OpenMatchTrainerProbe
import copy

import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

def data2boxes(data):
    boxes = []
    for i in range(len(data)):
        boxes_i = data[i]['instances'].gt_boxes.tensor
        if boxes_i.shape[0]:
            indices = i * torch.ones((boxes_i.shape[0], 1), dtype=boxes_i.dtype, device=boxes_i.device)
            boxes_i = torch.cat([indices, boxes_i], dim=1)
            boxes.append(boxes_i)
    if len(boxes):
        boxes = torch.cat(boxes, dim=0)
        return boxes
    else:
        return None

def instances2boxes(instances):
    boxes = []
    for i in range(len(instances)):
        boxes_i = instances[i].pred_boxes.tensor
        if boxes_i.shape[0]:
            indices = i * torch.ones((boxes_i.shape[0], 1), dtype=boxes_i.dtype, device=boxes_i.device)
            boxes_i = torch.cat([indices, boxes_i], dim=1)
            boxes.append(boxes_i)
    if len(boxes):
        boxes = torch.cat(boxes, dim=0)
        return boxes
    else:
        return None
    
def data2labels(data):
    labels = []
    for i in range(len(data)):
        labels_i = data[i]['instances'].gt_classes
        if labels_i.shape[0]:
            labels.append(labels_i)
    labels = torch.cat(labels, dim=0)
    return labels

def instances2labels(instances):
    labels = []
    for i in range(len(instances)):
        labels_i = instances[i].pred_classes
        if labels_i.shape[0]:
            labels.append(labels_i)
    labels = torch.cat(labels, dim=0)
    return labels

def locate_feature_roialign(feature_map, boxes, image_width, image_height):
    selected_features = []
    sx = feature_map.shape[3] / image_width
    sy = feature_map.shape[2] / image_height
    if len(boxes):
        boxes_level = torch.tensor(boxes, device=feature_map.device)
        boxes_level[:, 1] *= sx
        boxes_level[:, 2] *= sy
        boxes_level[:, 3] *= sx
        boxes_level[:, 4] *= sy
        selected_features_level = roi_align(feature_map, boxes_level, output_size=1, aligned=True)
        selected_features_level = torch.flatten(selected_features_level, start_dim=1)
        selected_features = selected_features_level
    else:
        selected_features = None
    return selected_features


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature    

    def forward(self, features, labels=None, mask=None, weights=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else: 
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        if weights is not None:
            loss = (loss.view(anchor_count, batch_size) * weights).sum() / weights.sum()
        else:
            loss = loss.view(anchor_count, batch_size).mean()
        return loss

# Adaptive Teacher Trainer
class ATeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)
        model = self.build_model(cfg)

        if comm.is_main_process():
            model.eval()

            # Use the same input size for AT and PCT.
            # If you want to follow the training resized target image size:
            H, W = 600, 1200

            # If you want to follow test-time resized Cityscapes-like input,
            # you may instead use:
            # H, W = 667, 1333

            device = next(model.parameters()).device
            input_tensor = torch.randn(1, 3, H, W).to(device)

            class DetectorInferenceWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    # x: [1, 3, H, W]
                    images = ImageList.from_tensors([x[0]])

                    # Inference detector path: backbone -> RPN -> RoI heads
                    features = self.model.backbone(images.tensor)
                    proposals, _ = self.model.proposal_generator(images, features)
                    outputs, _ = self.model.roi_heads(images, features, proposals)

                    # Return a tensor for tracing compatibility.
                    # The exact output is not important for FLOPs counting.
                    if isinstance(outputs, list) and len(outputs) > 0:
                        if hasattr(outputs[0], "pred_boxes"):
                            return outputs[0].pred_boxes.tensor.sum()
                        elif hasattr(outputs[0], "proposal_boxes"):
                            return outputs[0].proposal_boxes.tensor.sum()

                    return images.tensor.sum() * 0.0

            wrapper = DetectorInferenceWrapper(model).to(device)
            wrapper.eval()

            with torch.no_grad():
                flops = FlopCountAnalysis(wrapper, input_tensor)
                total_flops = flops.total() / 1e9

            print("=" * 80)
            print(f"Inference FLOPs at input size {H}x{W}: {total_flops:.2f} G")
            print("Unsupported ops:", flops.unsupported_ops())
            print("Uncalled modules:", flops.uncalled_modules())
            print("=" * 80)

        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        optimizer = self.build_optimizer(cfg, model)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
      
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.scale_list = np.array(cfg.SEMISUPNET.SCALE_LIST)
        self.scale_checkpoints = np.array(cfg.SEMISUPNET.SCALE_STEPS)
        self.consistency_losses = ConsistencyLosses()
        
        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())
        
        # CMT configs
        self.contrastive = cfg.SEMISUPNET.CONTRASTIVE
        self.supconloss = SupConLoss(contrast_mode='one')
        if cfg.MODEL.BACKBONE.NAME == 'build_vgg_backbone':
            self.feature_levels = ['vgg1', 'vgg2', 'vgg3', 'vgg4']
        elif cfg.MODEL.BACKBONE.NAME == 'build_resnet_backbone':
            self.feature_levels = ['res2', 'res3', 'res4']
        else:
            raise NotImplementedError

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume:
            self.start_iter = checkpoint.get("iteration", -1) + 1
        # if resume and self.checkpointer.has_checkpoint():
        #     self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                pass
                # self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)
        
    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseudo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
        
        elif proposal_type == "roih_stage_2":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.pred_boxes = new_boxes
            new_proposal_inst.pred_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data
    
    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        return label_list
    
    def get_images(self, label_data):
        images_list = []
        for label_datum in label_data:
            if "image" in label_datum.keys():
                images_list.append(copy.deepcopy(label_datum["image"]))
        return images_list

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================
    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
      
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

        data_time = time.perf_counter() - start

        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            
            label_data_q.extend(label_data_k)
            record_dict, _, _, _, _ = self.model(
                label_data_q, label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())
        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb.set_trace() 
            gt_unlabel_k = self.get_label(unlabel_data_k)

            #  0. remove unlabeled data labels
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            if self.cfg.STUDENT_SCALE:
                scale_mask=np.where(self.iter<self.scale_checkpoints)[0]
                if len(scale_mask)>0:
                    nstu_scale = self.scale_list[scale_mask[0]]
                else:
                    nstu_scale = 1.0
                self.stu_scale = np.random.normal(nstu_scale,0.15)
                if self.stu_scale < 0.4:
                    self.stu_scale = 0.4
                elif self.stu_scale > 1.0:
                    self.stu_scale = 1.0

                scaled_unlabel_data_k = [x.copy() for x in unlabel_data_k]
                img_k = scaled_unlabel_data_k[0]['image'].shape[1:]
                self.scale_k = T.Resize((int(img_k[0]*self.stu_scale), int(img_k[1]*self.stu_scale)))

                for item in scaled_unlabel_data_k:
                    item['image'] = item['image'].cuda()
                    item['image'] = self.scale_k(item['image'])
            else:
                # if student scaling is not used
                scaled_unlabel_data_k = [x.copy() for x in unlabel_data_k]

            #  1. generate the pseudo-label using teacher model
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                    features_teacher,
                    features_teacher_rs
                ) = self.model_teacher(unlabel_data_k, scaled_unlabel_data_k, branch="unsup_data_weak")

            #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k

            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            # 3. add pseudo-label to unlabeled data
            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )
            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q
            
            # 4. input both strongly and weakly augmented labeled data into student model
            record_all_label_data, _, _, _, _ = self.model(
                all_label_data, unlabel_data_q, branch="supervised"
            )
            record_dict.update(record_all_label_data)
     
            # 5. input strongly augmented unlabeled data into model
            if self.cfg.STUDENT_SCALE:
                scaled_unlabel_data_q = [x.copy() for x in unlabel_data_q]

                img_s = scaled_unlabel_data_q[0]['image'].shape[1:]
                self.scale_t = T.Resize((int(img_s[0]*self.stu_scale), int(img_s[1]*self.stu_scale)))
                for item in scaled_unlabel_data_q:
                    item['image'] = item['image'].cuda()
                    item['image']=self.scale_t(item['image'])
                    item['instances'].gt_boxes.scale(self.stu_scale,self.stu_scale)
                    gt_mask = item['instances'].gt_boxes.area()>16 
                    gt_boxes = item['instances'].gt_boxes[gt_mask]
                    gt_classes = item['instances'].gt_classes[gt_mask]
                    scores = item['instances'].scores[gt_mask]
                    item['instances'] = Instances(item['image'].shape[1:], gt_boxes=gt_boxes, gt_classes = gt_classes, scores=scores)
            else:
                scaled_unlabel_data_q = [x.copy() for x in unlabel_data_q] 

            (pseudo_losses, 
            proposals_into_roih, 
            rpn_stu,
            roi_stu,
            pred_idx,
            features_student, 
            features_student_qy)= self.model(
                scaled_unlabel_data_q, unlabel_data_q, branch="consistency_target"
            )
            new_pseudo_losses = {}
            for key in pseudo_losses.keys():
                new_pseudo_losses[key + "_pseudo"] = pseudo_losses[
                    key
                ]
            record_dict.update(new_pseudo_losses)
            
            if self.cfg.STUDENT_SCALE:
                stu_resized_proposals = []
                for k,proposals in enumerate(proposals_into_roih):
                    stu_resized_proposals.append(Instances(scaled_unlabel_data_q[0]['image'].shape[1:],
                                            proposal_boxes = proposals.proposal_boxes.clone(),
                                            objectness_logits = proposals.objectness_logits,
                                            gt_classes = proposals.gt_classes,
                                            gt_boxes = proposals.gt_boxes))
                    stu_resized_proposals[k].proposal_boxes.scale(1/self.stu_scale,1/self.stu_scale)
                proposals_into_roih=stu_resized_proposals
            
            with torch.no_grad():
                (_,
                _,
                roi_teach,
                _
                )= self.model_teacher(
                    unlabel_data_k, 
                    unlabel_data_k, 
                    branch="unsup_data_consistency", 
                    given_proposals=proposals_into_roih, 
                    proposal_index=pred_idx
                )
            
            cons_loss = self.consistency_losses.losses(roi_stu,roi_teach)
            record_dict.update(cons_loss)

            # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            for i_index in range(len(unlabel_data_k)):
                for k, v in unlabel_data_k[i_index].items():
                    label_data_k[i_index][k + "_unlabeled"] = v
            all_domain_data = label_data_k
            record_all_domain_data, _, _, _, _ = self.model(all_domain_data, label_data_k, branch="domain")
            record_dict.update(record_all_domain_data)
                
            # 7. object-level contrastive learning
            if self.contrastive:
                for inst in roi_teach:
                    full_scores = inst.full_scores  # shape: [num_proposals, num_classes]
                    scores = full_scores.max(dim=1).values  # shape: [num_proposals]
                    inst.scores = scores
        
                pesudo_roi_teach, _ = self.process_pseudo_label(
                    roi_teach, cur_threshold, "roih_stage_2", "thresholding"
                )
                joint_proposal_dict["proposals_pseudo_stage_2_roih"] = pesudo_roi_teach
                scaled_pesudo_roi_teach = []
                for inst in pesudo_roi_teach:
                    new_inst = Instances(inst.image_size)  
                    new_inst.pred_boxes = Boxes(inst.pred_boxes.tensor.clone())
                    new_inst.pred_classes = inst.pred_classes.clone()
                    # new_inst.full_scores = inst.full_scores.clone()
                    new_inst.scores = inst.scores.clone()
                    scaled_pesudo_roi_teach.append(new_inst)

                if self.cfg.STUDENT_SCALE:
                    stu_resized_proposals_stage_2 = []
                    filtered_indices_all = []  

                    for k,rois in enumerate(scaled_pesudo_roi_teach):
                        new_inst = Instances(scaled_unlabel_data_q[0]['image'].shape[1:])
                        new_boxes = rois.pred_boxes.clone()
                        new_boxes.scale(self.stu_scale, self.stu_scale)

                        areas = new_boxes.area()
                        valid_mask = areas > 16
                        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

                        filtered_boxes = new_boxes[valid_indices]
                        filtered_classes = rois.pred_classes[valid_indices]
                        filtered_scores = rois.scores[valid_indices]

                        new_inst.pred_boxes = filtered_boxes
                        new_inst.pred_classes = filtered_classes
                        new_inst.scores = filtered_scores
                        
                        stu_resized_proposals_stage_2.append(new_inst)
                        filtered_indices_all.append(valid_indices)

                    scaled_pesudo_roi_teach=stu_resized_proposals_stage_2
                
                joint_proposal_dict["scaled_proposals_pseudo_stage_2_roih"] = scaled_pesudo_roi_teach
                
                pesudo_roi_teach_filtered = []
                for rois, valid_indices in zip(pesudo_roi_teach, filtered_indices_all):
                    filtered_inst = Instances(rois.image_size)
                    filtered_inst.pred_boxes = Boxes(rois.pred_boxes.tensor[valid_indices])
                    filtered_inst.pred_classes = rois.pred_classes[valid_indices]
                    filtered_inst.scores = rois.scores[valid_indices]
                    pesudo_roi_teach_filtered.append(filtered_inst)

                pesudo_roi_teach = pesudo_roi_teach_filtered
                joint_proposal_dict["proposals_pseudo_stage_2_roih"] = pesudo_roi_teach

                boxes = instances2boxes(joint_proposal_dict["proposals_pseudo_stage_2_roih"])
                boxes_scaled = instances2boxes(joint_proposal_dict["scaled_proposals_pseudo_stage_2_roih"])

                image_width = all_unlabel_data[0]['image'].shape[2]
                image_height = all_unlabel_data[0]['image'].shape[1]

                image_width_scaled = all_unlabel_data[0]['image'].shape[2] * self.stu_scale
                image_height_scaled = all_unlabel_data[0]['image'].shape[1] * self.stu_scale
                
                if boxes is not None:
                    flags = []
                    for i in range(boxes.shape[0]):
                        box_i = boxes[i].to(torch.int)
                        image_index = box_i[0]
                        x1 = box_i[1]
                        y1 = box_i[2]
                        x2 = box_i[3]
                        y2 = box_i[4]
                        image_q_patch = unlabel_data_q[image_index]['image'][:, y1:y2, x1:x2].to(torch.float)
                        image_k_patch = unlabel_data_k[image_index]['image'][:, y1:y2, x1:x2].to(torch.float)
                        diff = (image_q_patch - image_k_patch).absolute().flatten()
                        ratio = (diff > 40).sum() / diff.numel()
                        if ratio > 0.5:
                            flags.append(0)
                        else:
                            flags.append(1)
                else:
                    flags = [0]
                
                # import ipdb;
                # ipdb.set_trace()
                if sum(flags):
                    # build contrastive loss
                    for feature_level in self.feature_levels:

                        object_features_student = locate_feature_roialign(features_student[feature_level], boxes_scaled, image_width_scaled, image_height_scaled)

                        object_features_student_qy = locate_feature_roialign(features_student_qy[feature_level], boxes, image_width, image_height)

                        object_features_teacher = locate_feature_roialign(features_teacher[feature_level], boxes, image_width, image_height)

                        object_features_teacher_rs = locate_feature_roialign(features_teacher_rs[feature_level], boxes_scaled, image_width_scaled, image_height_scaled)

                        object_features_student = nn.functional.normalize(object_features_student, dim=1)
                        object_features_teacher = nn.functional.normalize(object_features_teacher, dim=1) 
                        object_features_student_qy = nn.functional.normalize(object_features_student_qy, dim=1)
                        object_features_teacher_rs = nn.functional.normalize(object_features_teacher_rs, dim=1)
                        
                        object_features_all = torch.stack([object_features_student, object_features_teacher_rs], dim=1)
                        object_features_all_add = torch.stack([object_features_student_qy, object_features_teacher], dim=1)

                        object_labels = instances2labels(joint_proposal_dict["proposals_pseudo_stage_2_roih"])

                        flags = [bool(x) for x in flags]
                        object_features_all = object_features_all[flags]
                        object_labels = object_labels[flags]
                        object_features_all_add = object_features_all_add[flags]

                        object_features_all_total = torch.cat((object_features_all, object_features_all_add), dim=0) #torch.Size([24,2,128])
                        object_labels_total = torch.cat((object_labels, object_labels), dim=0) # torch.Size([24])

                        loss_contrastive_object = self.supconloss(object_features_all_total, object_labels_total)

                        # record contrastive loss
                        record_dict['loss_contrastive_object' + '_' + feature_level] = loss_contrastive_object * self.cfg.SEMISUPNET.CONTRASTIVE_LOSS_WEIGHT    
                else:
                    for feature_level in self.feature_levels:
                        record_dict['loss_contrastive_object' + '_' + feature_level] = torch.tensor(0.0)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        loss_dict[key] = record_dict[key] * 0
                    elif key.endswith('loss_cls_pseudo'):
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.CONSISTENCY_LOSS_WEIGHT
                    elif key.endswith('loss_rpn_cls_pseudo'):
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    elif (
                        key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT 
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.SUP_LOSS_WEIGHT
            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))
   
        if comm.is_main_process():

            ret.append(BestCheckpointer(
                checkpointer=self.checkpointer,
                model_attr_name="model",
                metric_name="bbox_student/AP50",
                file_name="model_best_student",
                is_greater_better=True,
                json_log_path="best_metrics_student.json"
            ))

            ret.append(BestCheckpointer(
                checkpointer=self.checkpointer,
                model_attr_name="model_teacher",
                metric_name="bbox/AP50",
                file_name="model_best_teacher",
                is_greater_better=True,
                json_log_path="best_metrics_teacher.json"
            ))

            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
