#!/bin/bash
#SBATCH --job-name=abc_145
#SBATCH --output=logs/abc_145.out
#SBATCH --error=logs/abc_145.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1163 862 576 1583 532 386 716 1124 1271
