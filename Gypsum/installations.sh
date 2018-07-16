#!/usr/bin/env bash

sudo apt-get install git htop tmux
sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev openssl python3.5-venv

# Install Conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
export PATH=~/miniconda3/bin:$PATH

# Create virtual environment
conda create -n pytorch python=3.5 numpy cython matplotlib memory_profiler
source activate pytorch

# Install packages via conda
conda install pytorch-cpu torchvision -c pytorch
conda install ipyparallel
conda install scipy sklearn
conda install tabulate xlwt pyyaml python-dateutil

# ---------------------------------------------------------------------------

# ========= PYTHON ============
#ps aux | grep -i '__main__.py *' | awk '{print $2}' | xargs kill -9


#cd ~
#wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
#sudo tar xzf Python-3.5.2.tgz
#cd Python-3.5.2
#sudo ./configure --with-ensurepip=install --prefix=/data/
#sudo make altinstall


# =========== CUDA ================

#export PYTHONPATH="../"
#export PYTHONPATH=$PYTHONPATH:"../"
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
#export LD_PRELOAD="/usr/lib/libtcmalloc.so.4"


# ============== Tensorflow ==========================


# pip install tensorflow-gpu
# pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl

# sudo apt-get install google-perftools

# Check supported whl formats for the machine
# import pip
# print(pip.pep425tags.get_supported())

# Get list of all whls
# curl -s https://storage.googleapis.com/tensorflow |xmllint --format - |grep whl

# Find the relevant 1.3.0 whl and install
# pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl


# GS
# pip install virtualenv
# virtualenv -p /usr/bin/python2.7 Virtualenvs/py2
# source Virtualenvs/py2/bin/activate
# ..
# pip install networkx==1.11
# pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl
#
# https://stackoverflow.com/questions/42013316/after-building-tensorflow-from-source-seeing-libcudart-so-and-libcudnn-errors




# pip3 install —upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0rc2-cp27-none-linux_x86_64.whl
