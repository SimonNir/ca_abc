#!/bin/bash
#SBATCH --job-name=abc_132
#SBATCH --output=logs/abc_132.out
#SBATCH --error=logs/abc_132.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1137 932 1577 176 338 1228 704 375 1506
