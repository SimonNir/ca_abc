#!/bin/bash
#SBATCH --job-name=abc_91
#SBATCH --output=logs/abc_91.out
#SBATCH --error=logs/abc_91.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 68 959 115 161 1684 1838 1148 609 1070 1668 1927 1921 1453 199 651 1167 565 1368 341 436
