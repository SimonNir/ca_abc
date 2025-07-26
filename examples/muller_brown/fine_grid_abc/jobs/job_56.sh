#!/bin/bash
#SBATCH --job-name=abc_56
#SBATCH --output=logs/abc_56.out
#SBATCH --error=logs/abc_56.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1704 925 1885 1116 217 1755 955 1888 1033 1655 14 569 1462 1728 1230 808 510 773 1584 1866
