#!/bin/bash
#SBATCH --job-name=abc_22
#SBATCH --output=logs/abc_22.out
#SBATCH --error=logs/abc_22.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 363 33 1034 759 838 198 1113 1159 583 696 658 36 785
