#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=64G

#SBATCH -p burst

#SBATCH -A birthright

#SBATCH --job-name=mb_sweep

#SBATCH --mail-user=nirenbergsd@ornl.gov
#SBATCH --mail-type=END

#SBATCH -o new_sweep.out

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u new_sweep.py > new_sweep.txt
