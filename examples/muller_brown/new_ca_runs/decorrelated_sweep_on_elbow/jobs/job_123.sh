#!/bin/bash
#SBATCH --job-name=abc_123
#SBATCH --output=logs/abc_123.out
#SBATCH --error=logs/abc_123.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3398 2733 3906 1597 2723 4539 967 4157 3293 725 802 3844 126 2474 1831 4005 4798 3894 1752 354 4398 55 2097 626 804 397
