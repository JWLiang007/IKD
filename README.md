# Exploring Inconsistent Knowledge Distillation for Object Detection with Data Augmentation 

This repository contains the official PyTorch implementation of the following paper at **ACMMM 2023**: 

> **Exploring Inconsistent Knowledge Distillation for Object Detection with Data Augmentation**<br>
> Jiawei Liang, Siyuan Liang, Aishan Liu, Ke Ma, Jingzhi Li, Xiaochun Cao<br>
> [https://arxiv.org/abs/2209.09841](https://arxiv.org/abs/2209.09841)

# Installation

* Install python (python == 3.8)
* Install pytorch (pytorch == 2.0.0)
* [Install mmcv_full (mmcv_full == 1.6.0)](https://github.com/open-mmlab/mmcv#installation)
* [Install mmdetection (mmdetection == 2.25.0) from source code](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)
  ```bash
    git clone https://github.com/JWLiang007/IKD.git
    cd IKD/
    pip install -r requirements/optional.txt
    pip install -v -e .
  ```

# Download Dataset and Checkpoint
* Download MS COCO2017 dataset 
* Unzip COCO dataset into data/coco/ in mmdetection/
* Download pretrained teacher model [retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth)
 from the repository of mmdetection 
* Put the downloaded pretrained model into checkpoints/ in mmdetection/


# Generate Adversarial Examples
```bash
# single GPU
python tools/ta_GT.py configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py  checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth --method difgsm  --show-dir data/adv_rtn_coco_8_5 --gen_adv_aug --eps 8 --alpha 2 --steps 5
# multi GPU
bash tools/dist_adv.sh  configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py  checkpoints/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth  8  --method difgsm  --show-dir data/adv_rtn_coco_8_5 --gen_adv_aug --eps 8 --alpha 2 --steps 5
```

## Train

```bash
#single GPU

# Step 1: train with DFA 
python tools/train.py configs/fgd/DFA_fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py
# Step 2: resume from epoch 16 and train without DFA
python tools/train.py configs/fgd/fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py --resume-from work_dirs/DFA_fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco/epoch_16.pth

#multi GPU

# Step 1: train with DFA 
bash tools/dist_train.sh configs/fgd/DFA_fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py 8
# Step 2: resume from epoch 16 and train without DFA
bash tools/dist_train.sh configs/fgd/fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py 8 --resume-from work_dirs/DFA_fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco/epoch_16.pth

```


## Test
```bash
#single GPU
python tools/test.py configs/fgd/DFA_fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py $PATH_CHECKPOINT --eval bbox

#multi GPU
bash tools/dist_test.sh configs/fgd/DFA_fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py $PATH_CHECKPOINT 8 --eval bbox
```

## Generalizability

#### Backdoor Defense 

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Index</th>
<th valign="center">Method</th>
<th valign="center">ASR</th>
<th valign="center">ASR Drop</th>


<tr>
<td align="center">1</td>
<td align="center">Victim</td>
<td align="center">96.7</td>
<td align="center">-</td>
</tr>

<tr>
<td align="center">2</td>
<td align="center">NAD</td>
<td align="center">82.88</td>
<td align="center">13.82</td>
</tr>
<tr>
<td align="center">3</td>
<td align="center">Ours</td>
<td align="center">78.26</td>
<td align="center">18.44(↑33%)</td>
</tr>

</tbody></table>

For more recent progress in backdoor defense, please refers to the following repo:

[https://github.com/JWLiang007/BD_DeCLIP.git](https://github.com/JWLiang007/BD_DeCLIP.git) 

and switches to the __bd__ branch.

## Acknowledgement

Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).
