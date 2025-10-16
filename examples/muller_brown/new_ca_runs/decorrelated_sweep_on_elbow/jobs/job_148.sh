#!/bin/bash
#SBATCH --job-name=abc_148
#SBATCH --output=logs/abc_148.out
#SBATCH --error=logs/abc_148.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4185 3703 2495 2651 4571 2552 402 3075 5055 4835 4656 1957 565 2776 3344 4870 1958 532 2522 2623 760 2311 4252 740 3523 5114
