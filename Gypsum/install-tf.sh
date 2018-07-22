#!/bin/bash

# Modules
module initrm cudnn/5.0
module initadd cuda75 cudnn/5.1
# More Module commands here:
# https://www.chpc.utah.edu/presentations/images-and-pdfs/SLURM-modules.pdf

# Install Miniconda
# curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# chmod a+x Miniconda3-latest-Linux-x86_64.sh
# ./Miniconda3-latest-Linux-x86_64.sh

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
export PATH=~/miniconda3/bin:$PATH

rm Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda upgrade --all

conda create -n tf python=3.5 numpy matplotlib memory_profiler

echo -e "Installation completed! \nrun \"source ~/.bashrc ; source activate tf\" before submitting TensorFlow jobs."
