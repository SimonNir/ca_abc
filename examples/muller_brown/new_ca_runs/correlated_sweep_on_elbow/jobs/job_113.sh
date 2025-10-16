#!/bin/bash
#SBATCH --job-name=abc_113
#SBATCH --output=logs/abc_113.out
#SBATCH --error=logs/abc_113.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 699 236 1560 1086 847 34 1 1006 194
