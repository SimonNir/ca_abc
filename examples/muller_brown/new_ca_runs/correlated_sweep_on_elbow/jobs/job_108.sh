#!/bin/bash
#SBATCH --job-name=abc_108
#SBATCH --output=logs/abc_108.out
#SBATCH --error=logs/abc_108.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1505 808 1178 748 371 133 1532 42 886
