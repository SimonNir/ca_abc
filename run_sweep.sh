#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=16G

#SBATCH -p burst

#SBATCH -A birthright

#SBATCH --job-name=mb_sweep

#SBATCH --mail-user=nirenbergsd@ornl.gov
#SBATCH --mail-type=END

#SBATCH -o new_sweep.out

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u sweep_v3_2.py > sweep_v3_2.txt
