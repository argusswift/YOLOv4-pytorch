# YOLOv4-pytorch

This is a PyTorch re-implementation of YOLOv4 architecture based on the official darknet implementation [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) with PASCAL VOC, COCO and Custom dataset
---
## Environment

* Nvida GeForce RTX 2070
* CUDA10.0
* CUDNN7.0
* windows
* python 3.6
```bash
# install packages
pip3 install -r requirements.txt --user
```
---
## Brief
* [x] Mish
* [x] Custom data
* [x] Data Augment (RandomHorizontalFlip, RandomCrop, RandomAffine, Resize)
* [x] Step lr Schedule 
* [x] Multi-scale Training (320 to 640)
* [x] focal loss
* [x] GIOU
* [x] Label smooth
* [x] Mixup
* [x] cosine lr
* [x] Multi-scale Test and Flip

---
## Reference

* tensorflow : https://github.com/Stinky-Tofu/Stronger-yolo
* pytorch : https://github.com/ultralytics/yolov3
https://github.com/Peterisfar/YOLOV3
* keras : https://github.com/qqwweee/keras-yolo3
