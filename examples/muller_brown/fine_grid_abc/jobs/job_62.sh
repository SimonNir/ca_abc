#!/bin/bash
#SBATCH --job-name=abc_62
#SBATCH --output=logs/abc_62.out
#SBATCH --error=logs/abc_62.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 25 1198 695 927 538 1119 1091 1617 431 1930 242 933 1915 1431 92 1961 166 1792 1332 7
