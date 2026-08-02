#!/bin/bash
#SBATCH --job-name=X
#SBATCH --partition=gpu
#SBATCH --account=project_rain_ottis
#SBATCH --gres=gpu:A100:1
#SBATCH --time=0-48:00:00
#SBATCH --mem=10GB
#SBATCH --output=X.log

module load rocky8-spack/master > /dev/null
module load cuda/11.3.1-gcc-10.3.0-ehi3 > /dev/null
module load py-pip/23.1.2-gcc-10.3.0-python-3.9.7-pmrk > /dev/null

python -m pip install --quiet numpy==1.23 > /dev/null
python -m pip install --quiet --user torch torchvision > /dev/null
#If torchvision does not install
#create METADATA file at /gpfs/mariana/home/semirs/.local/lib/python3.9/site-packages/numpy-1.23.0.dist-info/
python -m pip install git+https://github.com/Harry24k/MAIR.git > /dev/null
python -m pip install torchattacks > /dev/null
python -m pip install --upgrade torch torchvision > /dev/null

python x.py ResNet18_CIFAR10_STANDARD_0.report ResNet18_CIFAR10_STANDARD_1.report
