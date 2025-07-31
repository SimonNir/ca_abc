#!/bin/bash
#SBATCH --job-name=abc_98
#SBATCH --output=logs/abc_98.out
#SBATCH --error=logs/abc_98.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2583 4681 634 4250 3543 3734 2156 3612 3086 3893 1488 2383 4305 3206 3662 1678 2157 222 1878 637 3461 1190 2059 3619 3866 2049
