#!/bin/bash
#SBATCH --job-name=abc_36
#SBATCH --output=logs/abc_36.out
#SBATCH --error=logs/abc_36.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1563 760 100 1937 1064 1294 1871 1010 23 335 407 620 723 1030 1593 576 1842 979 1782 807
