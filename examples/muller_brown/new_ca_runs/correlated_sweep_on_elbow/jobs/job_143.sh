#!/bin/bash
#SBATCH --job-name=abc_143
#SBATCH --output=logs/abc_143.out
#SBATCH --error=logs/abc_143.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 836 65 912 560 191 1144 1356 1528 443
