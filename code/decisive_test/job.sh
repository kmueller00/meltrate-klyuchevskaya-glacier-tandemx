#!/bin/bash -l
#SBATCH --job-name=decisive
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/saturn/gwgi/gwgifu0h/00_libs/paper_checks/decisive/job.log
module load python/3.9-anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate TANDEMX
cd /home/saturn/gwgi/gwgifu0h/00_libs/paper_checks/decisive
env -u PYTHONPATH OMP_NUM_THREADS=8 python decisive.py
