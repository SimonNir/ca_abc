#!/bin/bash
#SBATCH --job-name=abc_33
#SBATCH --output=logs/abc_33.out
#SBATCH --error=logs/abc_33.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1088 1168 928 855 662 552 504 287 3 190 436 218 121
