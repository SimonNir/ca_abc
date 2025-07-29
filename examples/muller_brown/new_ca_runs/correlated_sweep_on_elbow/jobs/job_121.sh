#!/bin/bash
#SBATCH --job-name=abc_121
#SBATCH --output=logs/abc_121.out
#SBATCH --error=logs/abc_121.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 343 1038 1134 441 787 111 522 1591 427
