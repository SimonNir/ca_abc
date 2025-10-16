#!/bin/bash
#SBATCH --job-name=abc_142
#SBATCH --output=logs/abc_142.out
#SBATCH --error=logs/abc_142.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1190 1385 634 154 610 1101 218 374 167
