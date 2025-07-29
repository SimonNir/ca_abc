#!/bin/bash
#SBATCH --job-name=abc_131
#SBATCH --output=logs/abc_131.out
#SBATCH --error=logs/abc_131.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 80 365 776 1562 364 727 257 1240 544
