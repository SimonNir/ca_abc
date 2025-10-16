#!/bin/bash
#SBATCH --job-name=abc_169
#SBATCH --output=logs/abc_169.out
#SBATCH --error=logs/abc_169.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1214 434 2 903 107 951 296 668 336
