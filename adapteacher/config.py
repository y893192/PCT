# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ateacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True
    
    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = True
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128
    
    _C.SEMISUPNET.Trainer = "ateacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.DIS_TYPE = "res4"
    _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1
    _C.SEMISUPNET.CONSISTENCY_LOSS_WEIGHT = 0.3
    _C.SEMISUPNET.SCALE_LIST = (0.5, 0.6, 0.7,0.8,0.9,1.0)
    _C.SEMISUPNET.SCALE_STEPS = (87000,94000,101000,108000,115000,122000)
    _C.DOWN_SCALE = True
    _C.CONSISTENCY = True
    _C.SEMISUPNET.CONTRASTIVE = True
    _C.SEMISUPNET.CONTRASTIVE_LOSS_WEIGHT = 0.05

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True
