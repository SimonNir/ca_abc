#!/bin/bash
#SBATCH --job-name=abc_158
#SBATCH --output=logs/abc_158.out
#SBATCH --error=logs/abc_158.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4329 3976 2948 3581 4362 4025 3921 869 3504 3801 237 636 4023 4693 2124 4601 4967 792 701 1268 697 1101 3456 4322 4335 2508
