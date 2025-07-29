#!/bin/bash
#SBATCH --job-name=abc_120
#SBATCH --output=logs/abc_120.out
#SBATCH --error=logs/abc_120.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1035 911 196 1117 1115 685 1194 562 70
