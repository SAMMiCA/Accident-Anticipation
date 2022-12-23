# Accident-Anticipation

## Installation
**Note**: This repository is developed using 'pytorch 1.9.0' in Ubuntu 20.04 LTS OS with 'CUDA 11.3' GPU environment. However, more recent pytorch and CUDA versions are also compatible with this repository. 

a. Create a conda virtual environment of this repo and activate:

'''shell
conda create -n AA python=3.7 -y
conda activate AA
'''

b. Install official pytorch. 
'''shell
conda install pytorch==1.9.0 torchvision=0.10.0 cudatoolkit=11.3 -c pytorch
'''

c. Install the rest of the dependencies.
'''shell
pip install -r requirements.txt
'''

## Datasets

This repository supports for the down-sized version of [DADA-2000 dataset](https://github.com/JWFangit/LOTVS-DADA). Specifically, we reduced the image size at a half and trimmed the videos into accident clips with at most 450 frames. For more details, please refer to the code script `data/reduce_data.py`.

Currently this repository only contains the test dataset for trained agent.

## Testing

Run the testing
'''shell
bash script_MARC_3.sh test 0 4 marc
'''

