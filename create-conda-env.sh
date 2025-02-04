#!/usr/bin/env bash

export python_version="3.9"
export name="prosail"

conda create -n $name python=$python_version

# Enter virtualenv
conda activate $name
conda init bash

which python
python --version

conda install -y numpy pip pytest
# Installing Pytorch. Please change option for GPU use.
conda install -y pytorch torchvision cpuonly -c pytorch
conda install -y spyder
pip install -e .

# End
source deactivate

