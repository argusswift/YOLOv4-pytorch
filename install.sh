#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.6 ******************"
conda create -y --name $conda_env_name

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch with cuda10 ******************"
conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing pakages ******************"
pip install -r requirements.txt --user

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI


echo ""
echo ""
echo "****************** Installation complete! ******************"
