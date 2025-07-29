#!/bin/bash
#SBATCH --job-name=abc_68
#SBATCH --output=logs/abc_68.out
#SBATCH --error=logs/abc_68.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 762 940 1289 206 1164 456 1610 1492 955
