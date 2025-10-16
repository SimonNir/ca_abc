#!/bin/bash
#SBATCH --job-name=abc_137
#SBATCH --output=logs/abc_137.out
#SBATCH --error=logs/abc_137.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1323 421 591 1242 754 1046 51 1256 1475
