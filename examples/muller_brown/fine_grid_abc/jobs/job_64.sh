#!/bin/bash
#SBATCH --job-name=abc_64
#SBATCH --output=logs/abc_64.out
#SBATCH --error=logs/abc_64.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 311 1266 1737 1548 1424 1459 1246 1237 387 779 1156 1647 136 1508 1012 1977 179 674 1839 322
