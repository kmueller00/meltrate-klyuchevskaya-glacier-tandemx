#!/bin/bash -l
#SBATCH --job-name=masks_v2
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=/home/saturn/gwgi/gwgifu0h/00_libs/paper_checks/decisive/v2/masks.log
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
module load python/3.9-anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate TANDEMX
cd /home/saturn/gwgi/gwgifu0h/00_libs/paper_checks/decisive/v2
env -u PYTHONPATH python build_masks.py
