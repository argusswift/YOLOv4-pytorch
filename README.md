# YOLOv4-pytorch
This is a PyTorch re-implementation of YOLOv4 architecture based on the official darknet implementation [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) with PASCAL VOC, COCO and Custom dataset

![results](https://github.com/argusswift/YOLOv4-pytorch/blob/master/data/results.jpg)

## Highlights

### YOLOv4 with some useful module
This repo is simple to use,easy to read and uncomplicated to improve compared with others!!!

---
## Environment

* Nvida GeForce RTX 2070
* CUDA10.0
* CUDNN7.0
* windows or linux
* python 3.6

---
## Brief
* [x] Mish
* [x] Custom data
* [x] Data Augment (RandomHorizontalFlip, RandomCrop, RandomAffine, Resize)
* [x] Multi-scale Training (320 to 640)
* [x] focal loss
* [x] GIOU
* [x] Label smooth
* [x] Mixup
* [x] cosine lr

---
## Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```YOLOv4-pytorch```).  
```bash
pip3 install -r requirements.txt --user
```
**Note:** The install script has been tested on an Ubuntu 18.04 system and Window 10. In case of issues, check the [detailed installation instructions](INSTALL.md). 

## Prepared work

### 1、Git clone YOLOv4 repository
```Bash
git clone github.com/argusswift/YOLOv4-pytorch.git
```
update the `"PROJECT_PATH"` in the config/yolov4_config.py.

---

### 2、Download dataset
* Download Pascal VOC OR COCO dataset : 
For PASCAL VOC{[VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)、[VOC2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)}、
For COCO{[train2017_img](http://images.cocodataset.org/zips/train2017.zip)
、[train2017_ann](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
、[val2017_img](http://images.cocodataset.org/zips/val2017.zip)
 、[val2017_ann](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)
、[test2017_img](http://images.cocodataset.org/zips/test2017.zip)
 、[test2017_list](http://images.cocodataset.org/annotations/image_info_test2017.zip)
}

put them in the dir, and update the `"DATA_PATH"` in the params.py.
* Convert data format :use utils/voc.py or utils/coco.py convert the pascal voc *.xml format (COCO *.json format)to *.txt format (Image_path0 &nbsp; xmin0,ymin0,xmax0,ymax0,class0 &nbsp; xmin1,ymin1...)

### 3、Download weight file
* Darknet pre-trained weight :  [darknet53-448.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

Make dir `weight/` in the YOLOv4 and put the weight file in.

---
## To train

Run the following command to start training and see the details in the `config/yolov4_config.py`
```Bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --weight_path weight/darknet53_448.weights --gpu_id 0 > nohup.log 2>&1 &
```
Also * It supports to resume training adding `--resume`, it will load `last.pt` automaticly by using commad
```Bash
WEIGHT_PATH=weight/last.pt

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --weight_path $WEIGHT_PATH --gpu_id 0 > nohup.log 2>&1 &

```
---
## To detect
modify your detecte img path:DATA_TEST=/path/to/your/test_data # your own images
```Bash
for VOC dataset:
CUDA_VISIBLE_DEVICES=0 python3 val_voc.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval --mode det
for COCO dataset:
CUDA_VISIBLE_DEVICES=0 python3 val_coco.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval --mode det
```
The images can be seen in the `output/`

---
## To evaluate
modify your evaluate dataset path:DATA_TEST=/path/to/your/test_data # your own images
```Bash
for VOC dataset:
CUDA_VISIBLE_DEVICES=0 python3 val_voc.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval --mode val
for COCO dataset:
CUDA_VISIBLE_DEVICES=0 python3 val_coco.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval --mode val
```
The images can be seen in the `output/`

---
## To visualize heatmaps
set showatt=Ture in val_voc.py and you will see the heatmaps emerged from network' output
```Bash
for VOC dataset:
CUDA_VISIBLE_DEVICES=0 python3 val_voc.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval
for COCO dataset:
CUDA_VISIBLE_DEVICES=0 python3 val_coco.py --weight_path weight/best.pt --gpu_id 0 --visiual $DATA_TEST --eval
```
The heatmaps can be seen in the `output/` like this:

![heatmaps](https://github.com/argusswift/YOLOv4-pytorch/blob/master/data/heatmap.jpg)
---
## Reference

* tensorflow : https://github.com/Stinky-Tofu/Stronger-yolo
* pytorch : https://github.com/ultralytics/yolov3、
https://github.com/Peterisfar/YOLOV3
* keras : https://github.com/qqwweee/keras-yolo3
