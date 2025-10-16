#!/bin/bash
#SBATCH --job-name=abc_137
#SBATCH --output=logs/abc_137.out
#SBATCH --error=logs/abc_137.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3806 102 4566 5073 2635 985 52 2273 1160 1504 3034 4136 4204 546 1120 4881 773 1791 4541 2361 4975 2307 3567 2644 3321 3902
