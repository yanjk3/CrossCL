# CrossCL

Official PyTorch implementation of ICME2023 paper “Self-supervised Cross-stage Regional Contrastive Learning for Object Detection”.

## News
- 2023/08/17 We apply CrossCL ResNet50 backbone to our sparse detector [ASAG](https://github.com/iSEE-Laboratory/ASAG) and improve +1.2 AP.
- 2023/07/26 Checkpoints and Logs are released at [here](https://pan.baidu.com/s/1aCacbdBBEolAwxtMNmD0RA), key: icme.
- 2023/04/02 Code released.

## Environments
- python 3.7
- pytorch 1.6.0
- cuda 10.2

## Pre-training
All the instructions for pre-training a ResNet50-FPN on ImageNet or COCO can be found in ./sh.

Please modify the path to the dataset according to your local path.

For example, to pre-train CrossCL on ImageNet for 200 epochs, run the following command:
```
bash sh/crosscl_200e_imagenet.sh
```
Please note that by default, the model is trained on 8 GPUs.

## Transferring to Object Detection
To transfer with [Detectron2](https://github.com/facebookresearch/detectron2), you should convert the pre-trained model to a standard R50-FPN model by running the following command:
```
python tools/convert-pretrain-to-detectron.py /path/to/input/checkpoint /path/to/output/checkpoint
```
Then, you can use the official Detectron2 to train the detection model. 

Note, you should first install Detectron2 and then place COCO2017 dataset to detection/datasets/coco (you may use a soft-link).

Finally, to train a Mask R-CNN on COCO for 1x schedule, run the following script:
```
cd detection
bash finetune_coco_1x.sh
```

To transfer with [mmdetection](https://github.com/open-mmlab/mmdetection), you should convert the pre-trained model to a standard R50 model by running the following command:
```
python tools/convert-pretrain-to-mmdetection.py /path/to/input/checkpoint /path/to/output/checkpoint
```
Afterwards, you can go ahead to use the official mmdetection to train the detection model.

Note, you should first install mmdetection. 

To train a Mask R-CNN on COCO for 1x schedule, run the following training script:
```
cd mmdetection
sh ./tools/dist_train.sh configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \ 
    --options model.init_cfg.checkpoint=/path/to/output/checkpoint 8
```
Note, we do not provide the training code for training with mmdetection.

However, you can use the converted backbone model and follow the guidelines of mmdetection to train the model.

## Checkpoints and Logs
The pre-training/converted checkpoints and the pre-training/fine-tunig training logs can be downloaded [here](https://pan.baidu.com/s/1aCacbdBBEolAwxtMNmD0RA), key: icme.

If you have any problem, plz feel free to open an issue or contact me.

## Acknowledgement 
- This repository is heavily based on [MoCo](https://github.com/facebookresearch/moco) and [ReSim](https://github.com/Tete-Xiao/ReSim).

- If you use this paper/code in your research, please consider citing us:
```
@inproceedings{yan2023cross,
  title={Self-supervised Cross-stage Regional Contrastive Learning for Object Detection},
  author={Yan, junkai and Yang, Lingxiao and Gao, Yipeng and Zheng, Wei-Shi},
  booktitle={ICME},
  year={2023},
}
```
