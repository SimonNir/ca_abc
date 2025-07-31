#!/bin/bash
#SBATCH --job-name=abc_39
#SBATCH --output=logs/abc_39.out
#SBATCH --error=logs/abc_39.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 5039 4946 2747 3371 625 3413 1787 4575 1799 2353 485 3403 4590 1856 3223 236 544 4938 4153 2770 3273 4176 1945 3472 344 1696
