#!/bin/bash
#SBATCH --job-name=abc_125
#SBATCH --output=logs/abc_125.out
#SBATCH --error=logs/abc_125.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1707 669 38 4181 3219 2502 4840 1004 362 5059 2272 4166 1621 4751 1594 1786 2007 2102 2280 738 4128 1758 46 3433 1444 4887
