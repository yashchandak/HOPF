#!/usr/bin/env bash

sudo apt-get install git htop tmux
sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev openssl python3.5-venv

cd ~
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
sudo tar xzf Python-3.5.2.tgz
cd Python-3.5.2
sudo ./configure --with-ensurepip=install --prefix=/data/
sudo make altinstall

cd ~
mkdir Virtualenvs
python3.5 -m venv Virtualenvs/tf-1.3-nx0.19
source Virtualenvs/tf-1.3-nx0.19/bin/activate
pip install --upgrade pip
pip install numpy
pip install pillow
pip install scipy
pip install scikit-learn
pip install tabulate
pip install xlwt
pip install python-dateutil
pip install matplotlib
pip install pyyaml
pip install networkx==1.9
pip install --upgrade
