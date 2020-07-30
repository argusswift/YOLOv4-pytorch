# Installation

This document contains detailed instructions for installing the necessary dependencies for YOLOv4-pytorch. The instrustions have been tested on an Ubuntu 18.04 system and windows 10. We recommend using the [install script](install.sh) if you have not already tried that.  

### Requirements  
* Conda installation with Python 3.6. If not already installed, install from https://www.anaconda.com/distribution/.
* Nvidia GPU.
* CUDA10.0
* CUDNN7.0

## Step-by-step instructions  
#### Create and activate a conda environment
```bash
conda create --name YOLOv4-pytorch python=3.6
conda activate YOLOv4-pytorch
```

#### Install PyTorch  
Install PyTorch with cuda10.0  
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

**Note:**  
- It is possible to use any PyTorch supported version of CUDA (not necessarily v10).   
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

#### Install numpy,opencv-python, tqdm, argparse, pickleshare and tensorboardX 
```bash
pip install -r requirements.txt --user
```


#### Install the coco toolkit  
If you want to use COCO dataset for training, install the coco python toolkit. You additionally need to install cython to compile the coco toolkit.
```bash
conda install cython
pip install pycocotools
or using
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
more information please see https://github.com/philferriere/cocoapi
```


