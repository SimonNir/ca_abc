#!/bin/bash
#SBATCH --job-name=abc_20_2
#SBATCH --output=logs/abc_20_2.out
#SBATCH --error=logs/abc_20_2.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 124 91
