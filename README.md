[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
# YOLOv4-pytorch (attentive YOLOv4 and Mobilenetv3 YOLOv4)

This is a PyTorch re-implementation of YOLOv4 architecture based on the official darknet implementation [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) with PASCAL VOC, COCO and Customer dataset

## Results(updating)

| name | train Dataset | test Dataset | test size | mAP | inference time(ms) | params(M) |model link |
| :----- | :----- | :------ | :----- | :-----| :-----|  :-----|:-----|
| mobilenetv2-YOLOV4 | VOC trainval(07+12) | VOC test(07) | 416 | 0.851| 11.29 | 46.34 | [args](https://pan.baidu.com/s/10cAzQLHZmPxpHyCsNncV_w) |

## Update!!!
Mobilenetv3-YOLOv4 is arriving!（You only need to change the MODEL_TYPE in config/yolov4_config.py）

## News!!!
This repo add some useful attention methods in backbone.The following pictures illustrate such thing:

* SEnet(CVPR 2017)

![SEnet](https://github.com/argusswift/YOLOv4-pytorch/blob/master/data/SEnet.jpg)

* CBAM(CVPR 2018)

![CBAM](https://github.com/argusswift/YOLOv4-pytorch/blob/master/data/CBAM.png)

## Highlights

### YOLOv4 (attentive YOLOv4 and Mobilenet-YOLOv4) with some useful module
This repo is simple to use,easy to read and uncomplicated to improve compared with others!!!

## Environment

* Nvida GeForce RTX 2080TI
* CUDA10.0
* CUDNN7.0
* windows or linux
* python 3.6

---
## Brief
* [x] DO-Conv([arxiv2020](https://arxiv.org/abs/2006.12030))(torch>=1.2)
* [x] Attention
* [x] fp_16 training
* [x] Mish
* [x] Custom data
* [x] Data Augment (RandomHorizontalFlip, RandomCrop, RandomAffine, Resize)
* [x] Multi-scale Training (320 to 640)
* [x] focal loss
* [x] CIOU
* [x] Label smooth
* [x] Mixup
* [x] cosine lr

---
## Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```YOLOv4-pytorch```).  
```bash
pip3 install -r requirements.txt --user
```
**Note:** The install script has been tested on an Ubuntu 18.04 and Window 10 system. In case of issues, check the [detailed installation instructions](INSTALL.md). 

## Prepared work

### 1、Git clone YOLOv4 repository
```Bash
git clone github.com/argusswift/YOLOv4-pytorch.git
```
Update the `"PROJECT_PATH"` in the config/yolov4_config.py.

---

### 2、Prepared dataset
### PascalVOC
  ```Shell
  # Download the data.
  cd $HOME/data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # Extract the data.
  tar -xvf VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```
* Download links:{[VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)、[VOC2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)}、
### MSCOCO 2017
  ```Shell
	#step1: download the following data and annotation
	2017 Train images [118K/18GB]
	2017 Val images [5K/1GB]
	2017 Test images [41K/6GB]
	2017 Train/Val annotations [241MB]
	#step2: arrange the data to the following structure
	COCO
 	---train
	---test
	---val
	---annotations
  ```
* Download links:{[train2017_img](http://images.cocodataset.org/zips/train2017.zip)
、[train2017_ann](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
、[val2017_img](http://images.cocodataset.org/zips/val2017.zip)
 、[val2017_ann](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)
、[test2017_img](http://images.cocodataset.org/zips/test2017.zip)
 、[test2017_list](http://images.cocodataset.org/annotations/image_info_test2017.zip)
}

### And then
* Put them in the dir, and update the `"DATA_PATH"` in the config/yolov4_config.py.
* (for COCO) Use coco_to_voc.py to transfer COCO datatype to VOC datatype.
* Convert data format :use utils/voc.py or utils/coco.py convert the pascal voc *.xml format (COCO *.json format)to *.txt format (Image_path &nbsp; xmin0,ymin0,xmax0,ymax0,class0 &nbsp; xmin1,ymin1,xmax1,ymax1,class1 &nbsp;...).


### 3、Download weight file
* Darknet pre-trained weight :  [yolov4](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)
* Mobilenet pre-trained weight :   [mobilenetv2](https://pan.baidu.com/s/1sjixK2L9L0YgQnvfDuVTJQ)(code:args),[mobilenetv3](https://pan.baidu.com/s/175wKejULuM0ZD05b9iSftg)(code:args)
* Make dir `weight/` in the YOLOv4 and put the weight file in.
* set MODEL_TYPE in config/yolov4_config.py when you run training program.

### 4、Transfer to your own dataset（train your own dataset）
* Put pictures of your dataset into the JPEGImages folder, and Annotations files into the Annotations folder.
* Use the xml_to_txt.py file to write the list of training and test files to ImageSets/Main/*.txt.
* Convert data format :use utils/voc.py or utils/coco.py convert the pascal voc *.xml format (COCO *.json format)to *.txt format (Image_path &nbsp; xmin0,ymin0,xmax0,ymax0,class0 &nbsp; xmin1,ymin1,xmax1,ymax1,class1 &nbsp;...).
---
## To train

Run the following command to start training and see the details in the `config/yolov4_config.py` and you should set DATA_TYPE is VOC or COCO when you run training program.
```Bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --weight_path weight/yolov4.weights --gpu_id 0 > nohup.log 2>&1 &
```
Also * It supports to resume training adding `--resume`, it will load `last.pt` automaticly by using commad
```Bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --weight_path weight/last.pt --gpu_id 0 > nohup.log 2>&1 &
```
---
## To detect
Modify your detecte img path:DATA_TEST=/path/to/your/test_data # your own images
```Bash
for VOC dataset:
CUDA_VISIBLE_DEVICES=0 python3 eval_voc.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval --mode det
for COCO dataset:
CUDA_VISIBLE_DEVICES=0 python3 eval_coco.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval --mode det
```
The images can be seen in the `output/`. you could see pictures like follows:

![det-result](https://github.com/argusswift/YOLOv4-pytorch/blob/master/data/det-result.jpg)

---
## To test video
Modify:
* video_path:/path/to/your/video
* weight_path:/path/to/your/weight
* output_dir:/path/to/save/dir
```Bash
CUDA_VISIBLE_DEVICES=0 python3 video_test.py --weight_path best.pt --gpu_id 0 --video_path video.mp4 --output_dir --output_dir
```
---
## To evaluate （PASCAL VOC）
Modify your evaluate dataset path:DATA_PATH=/path/to/your/test_data # your own images
```Bash
for VOC dataset:
CUDA_VISIBLE_DEVICES=0 python3 eval_voc.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval --mode val
```

![results](https://github.com/argusswift/YOLOv4-pytorch/blob/master/data/results.jpg)

If you want to see the picture above, you should use follow commands:

```Bash
# To get ground truths of your dataset
python3 utils/get_gt_txt.py
# To plot P-R curve and calculate mean average precision
python3 utils/get_map.py 
```

## To evaluate （COCO）
Modify your evaluate dataset path:DATA_PATH=/path/to/your/test_data # your own images
```bash
CUDA_VISIBLE_DEVICES=0 python3 eval_coco.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval --mode val

type=bbox
Running per image evaluation...      DONE (t=0.34s).
Accumulating evaluation results...   DONE (t=0.08s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.438
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.607
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.486
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.632
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.691
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.790
```
---
## To evaluate your model parameters
```Bash
python3 utils/modelsize.py
```
---
## To visualize heatmaps
Set showatt=Ture in val_voc.py and you will see the heatmaps emerged from network' output
```Bash
for VOC dataset:
CUDA_VISIBLE_DEVICES=0 python3 eval_voc.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval
for COCO dataset:
CUDA_VISIBLE_DEVICES=0 python3 eval_coco.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval
```
The heatmaps can be seen in the `output/` like this:

![heatmaps](https://github.com/argusswift/YOLOv4-pytorch/blob/master/data/heatmap.jpg)
---
 
## Reference

* tensorflow : https://github.com/Stinky-Tofu/Stronger-yolo
* pytorch : https://github.com/ultralytics/yolov3
https://github.com/Peterisfar/YOLOV3
* keras : https://github.com/qqwweee/keras-yolo3
