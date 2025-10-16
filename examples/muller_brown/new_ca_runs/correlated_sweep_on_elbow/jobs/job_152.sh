#!/bin/bash
#SBATCH --job-name=abc_152
#SBATCH --output=logs/abc_152.out
#SBATCH --error=logs/abc_152.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4 1319 614 1602 1007 244 1318 1156 1274
