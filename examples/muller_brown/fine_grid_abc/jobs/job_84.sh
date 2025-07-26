#!/bin/bash
#SBATCH --job-name=abc_84
#SBATCH --output=logs/abc_84.out
#SBATCH --error=logs/abc_84.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1344 1015 1791 1819 1303 265 191 1082 1419 1828 1021 865 1316 1903 1810 248 1697 442 145 1492
