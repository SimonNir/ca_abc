#!/bin/bash
#SBATCH --job-name=abc_25
#SBATCH --output=logs/abc_25.out
#SBATCH --error=logs/abc_25.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 727 823 435 1271 101 1315 475 878 1707 1353 1482 1052 904 832 1260 1528 566 1998 946 1111
