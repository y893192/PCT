# Refining Pseudo Labeling with Multi-Scale Consistency Matching for Contrastive Domain Adaptive Object Detection

By Yan Yuan, Xiaodong Wang, Lei Huang.

This is the implementation of our paper: **Refining Pseudo Labeling with Multi-Scale Consistency Matching for Contrastive Domain Adaptive Object Detection**.

> :exclamation: Note: The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.
> 
## the Framework of the Proposed Module
![Model Architecture](model.png)

## Preparation
- OS: CentOS 7
- Python: 3.9.21
- CUDA: 11.3
- PyTorch: 1.12.1
- Torchvision: 0.13.1
### Build Detectron2 from Source
Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) to install Detectron2. We use version: detectron2==0.5

## Dataset
Experiments on 4 image datasets: Cityscapes, Foggy Cityscapes, KITTI, Sim10k.
### Download the datasets
| # | Datasets | Download |
|:--:|:-------:|:--------:|
| 1  |Cityscapes | [Link](https://www.cityscapes-dataset.com/)   |
| 2  |Foggy Cityscapes| [Link](https://www.cityscapes-dataset.com/)   |
| 3  |KITTI | [Link](https://www.cvlibs.net/datasets/kitti/)   |
| 4  |Sim10k | [Link](https://fcav.engin.umich.edu/projects/driving-in-the-matrix)   |
### Organize the dataset as the Cityscapes and PASCAL VOC format following:
```text  
pct/
└── datasets/
    ├── cityscapes/
    │   ├── gtFine/
    │   │   ├── train/
    │   │   ├── test/
    │   │   └── val/
    │   └── leftImg8bit/
    │       ├── train/
    │       ├── test/
    │       └── val/
    ├── cityscapes_foggy/
    │   ├── gtFine/
    │   │   ├── train/
    │   │   ├── test/
    │   │   └── val/
    │   └── leftImg8bit/
    │       ├── train/
    │       ├── test/
    │       └── val/
    ├── kitti/
    │   ├── Annotations/
    │   ├── ImageSets/
    │   └── JPEGImages/
    └── sim/
        ├── Annotations/
        ├── ImageSets/
        └── JPEGImages/
```

## Training
``` python
python train_net.py \
      --num-gpus 1 \
      --config configs/faster_rcnn_VGG_cross_city.yaml \
      OUTPUT_DIR outputs/c2f
```

## Resume the training
``` python
python train_net.py \
      --resume \
      --num-gpus 1 \
      --config configs/faster_rcnn_VGG_cross_city.yaml MODEL.WEIGHTS <your weight>.pth
```

## Evaluation
``` python
python train_net.py \
      --eval-only \
      --num-gpus 1 \
      --config configs/faster_rcnn_VGG_cross_city.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

## Main Results

## Acknowledgment
We are very grateful for these excellent works: [AT](https://github.com/facebookresearch/adaptive_teacher), [CMT](https://github.com/Shengcao-Cao/CMT/tree/main/CMT_AT), [2PCNet](https://github.com/mecarill/2pcnet), [MoCo](https://github.com/facebookresearch/moco), [Detectron2](https://github.com/facebookresearch/detectron2). Please follow their respective licenses for usage and redistribution. Thanks for their awesome works.

## Contact
Feel free to contact me if there is any question. (yuanyan@stu.ouc.edu.cn)


