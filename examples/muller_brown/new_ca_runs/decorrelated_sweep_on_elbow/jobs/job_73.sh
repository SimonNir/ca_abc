#!/bin/bash
#SBATCH --job-name=abc_73
#SBATCH --output=logs/abc_73.out
#SBATCH --error=logs/abc_73.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1246 925 3726 4254 1640 221 2114 3233 4473 3187 4893 979 776 4960 1885 115 3339 2865 4318 829 282 4728 4805 3061 130 3040
