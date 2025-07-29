#!/bin/bash
#SBATCH --job-name=abc_48
#SBATCH --output=logs/abc_48.out
#SBATCH --error=logs/abc_48.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1211 123 6 309 97 1561 1171 1408 1311
