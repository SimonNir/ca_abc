#!/bin/bash
#SBATCH --job-name=abc_33
#SBATCH --output=logs/abc_33.out
#SBATCH --error=logs/abc_33.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1877 1371 908 1679 1554 524 836 1727 0 928 1357 1017 1790 158 279 1454 251 1330 1013 663
