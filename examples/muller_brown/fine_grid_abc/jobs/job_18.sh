#!/bin/bash
#SBATCH --job-name=abc_18
#SBATCH --output=logs/abc_18.out
#SBATCH --error=logs/abc_18.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1642 830 380 1582 1856 123 817 629 493 818 62 1277 1694 137 15 119 1224 187 1218 1083
