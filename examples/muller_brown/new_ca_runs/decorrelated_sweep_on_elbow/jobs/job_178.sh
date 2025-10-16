#!/bin/bash
#SBATCH --job-name=abc_178
#SBATCH --output=logs/abc_178.out
#SBATCH --error=logs/abc_178.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2854 1754 2567 4392 1018 2575 26 4532 3307 3302 1804 3675 248 449 1559 1704 1896 4200 4832 1884 3130 5008 300 3669 3089 3348
