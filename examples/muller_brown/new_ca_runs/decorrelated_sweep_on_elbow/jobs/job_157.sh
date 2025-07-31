#!/bin/bash
#SBATCH --job-name=abc_157
#SBATCH --output=logs/abc_157.out
#SBATCH --error=logs/abc_157.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3805 1369 2021 2491 4215 1830 4723 3195 2867 3838 4370 2718 4275 2244 273 454 781 4582 1034 30 730 3329 1351 3038 1217 2936
