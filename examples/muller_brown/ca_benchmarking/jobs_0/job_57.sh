#!/bin/bash
#SBATCH --job-name=abc_57
#SBATCH --output=logs/abc_57.out
#SBATCH --error=logs/abc_57.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2 450 1126 1116 1093 912 722 248 396 1079 341 12 736
