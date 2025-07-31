#!/bin/bash
#SBATCH --job-name=abc_83
#SBATCH --output=logs/abc_83.out
#SBATCH --error=logs/abc_83.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1870 3424 3069 4968 3209 771 1097 3085 3277 3280 2396 2624 2454 3336 143 4677 838 265 4113 832 5074 5047 4363 1904 1009 4729
