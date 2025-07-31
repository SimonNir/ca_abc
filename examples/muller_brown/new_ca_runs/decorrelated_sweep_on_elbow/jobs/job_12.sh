#!/bin/bash
#SBATCH --job-name=abc_12
#SBATCH --output=logs/abc_12.out
#SBATCH --error=logs/abc_12.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2300 3513 3727 3447 1472 2365 1015 1490 2122 1876 2433 3536 4476 3830 2861 1586 1063 4614 206 4340 4064 164 1127 3261 1156 3043
