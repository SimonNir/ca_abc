#!/bin/bash
#SBATCH --job-name=abc_60
#SBATCH --output=logs/abc_60.out
#SBATCH --error=logs/abc_60.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 22 1041 880 1235 675 1343 1592 1734 182 1239 1300 752 901 415 1434 1081 1954 313 1263 924
