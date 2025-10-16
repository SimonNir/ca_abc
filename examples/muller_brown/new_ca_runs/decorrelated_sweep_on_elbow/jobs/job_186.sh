#!/bin/bash
#SBATCH --job-name=abc_186
#SBATCH --output=logs/abc_186.out
#SBATCH --error=logs/abc_186.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4562 3068 643 472 3126 4620 3766 1443 3896 2420 389 172 3425 3390 5093 2475 2201 2068 1598 3960 3028 2016 3342 2960 51 271
