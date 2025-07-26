#!/bin/bash
#SBATCH --job-name=abc_24
#SBATCH --output=logs/abc_24.out
#SBATCH --error=logs/abc_24.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 366 1706 1610 1918 583 163 1539 1171 1232 1712 222 1084 1467 1841 39 195 1254 1929 260 1035
