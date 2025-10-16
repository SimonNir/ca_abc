#!/bin/bash
#SBATCH --job-name=abc_125
#SBATCH --output=logs/abc_125.out
#SBATCH --error=logs/abc_125.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 321 894 344 648 1619 1426 390 1549 209
