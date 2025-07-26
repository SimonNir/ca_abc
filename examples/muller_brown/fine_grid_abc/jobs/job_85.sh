#!/bin/bash
#SBATCH --job-name=abc_85
#SBATCH --output=logs/abc_85.out
#SBATCH --error=logs/abc_85.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1488 1066 394 1408 1058 445 1599 1947 876 269 917 316 1423 561 1523 27 1702 382 1980 505
