#!/bin/bash
#SBATCH --job-name=abc_122
#SBATCH --output=logs/abc_122.out
#SBATCH --error=logs/abc_122.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 871 723 1140 702 948 12 1367 1333 505
