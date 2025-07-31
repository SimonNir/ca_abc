#!/bin/bash
#SBATCH --job-name=abc_1
#SBATCH --output=logs/abc_1.out
#SBATCH --error=logs/abc_1.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2601 2631 3497 3883 3541 984 3438 583 1702 2726 25 4141 878 401 4295 4912 299 4650 3736 3762 2716 3420 4156 1239 4393 1622
