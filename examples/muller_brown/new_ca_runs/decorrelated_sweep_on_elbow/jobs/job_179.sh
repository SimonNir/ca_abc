#!/bin/bash
#SBATCH --job-name=abc_179
#SBATCH --output=logs/abc_179.out
#SBATCH --error=logs/abc_179.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 850 4149 2759 3013 4537 230 677 3780 2070 4015 3356 4544 2301 909 1053 3449 2851 255 3913 2118 3102 3064 3495 1404 2348 691
