#!/bin/bash
#SBATCH --job-name=abc_139
#SBATCH --output=logs/abc_139.out
#SBATCH --error=logs/abc_139.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 177 1381 412 983 814 796 313 612 1578
