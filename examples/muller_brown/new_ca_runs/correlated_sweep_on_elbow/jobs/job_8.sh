#!/bin/bash
#SBATCH --job-name=abc_8
#SBATCH --output=logs/abc_8.out
#SBATCH --error=logs/abc_8.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 376 1064 122 1354 452 118 1262 94 743
