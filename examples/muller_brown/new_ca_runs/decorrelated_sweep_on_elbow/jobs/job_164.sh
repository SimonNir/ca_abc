#!/bin/bash
#SBATCH --job-name=abc_164
#SBATCH --output=logs/abc_164.out
#SBATCH --error=logs/abc_164.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4093 3816 1516 5079 3283 3263 5082 2998 2657 1213 875 5049 286 258 2553 1792 4289 864 1932 2467 1514 1084 1800 2917 2696 956
