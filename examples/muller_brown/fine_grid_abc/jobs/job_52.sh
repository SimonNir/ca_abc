#!/bin/bash
#SBATCH --job-name=abc_52
#SBATCH --output=logs/abc_52.out
#SBATCH --error=logs/abc_52.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1713 1329 987 1808 1337 1945 1131 1740 1715 180 1933 1858 386 1524 207 1564 1607 1407 1575 1387
