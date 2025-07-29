#!/bin/bash
#SBATCH --job-name=abc_151
#SBATCH --output=logs/abc_151.out
#SBATCH --error=logs/abc_151.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 874 1231 822 635 413 710 220 1514 989
