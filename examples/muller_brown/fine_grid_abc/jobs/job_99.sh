#!/bin/bash
#SBATCH --job-name=abc_99
#SBATCH --output=logs/abc_99.out
#SBATCH --error=logs/abc_99.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1240 897 1834 777 951 532 1733 1979 1339 1180 1256 381 1485 1350 170 1905 110 1193 1141 1376
