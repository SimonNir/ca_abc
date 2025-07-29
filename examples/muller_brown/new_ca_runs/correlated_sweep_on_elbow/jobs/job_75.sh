#!/bin/bash
#SBATCH --job-name=abc_75
#SBATCH --output=logs/abc_75.out
#SBATCH --error=logs/abc_75.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 679 400 718 398 1136 298 864 520 909
