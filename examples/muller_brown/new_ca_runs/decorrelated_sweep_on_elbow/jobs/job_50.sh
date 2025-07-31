#!/bin/bash
#SBATCH --job-name=abc_50
#SBATCH --output=logs/abc_50.out
#SBATCH --error=logs/abc_50.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 593 5081 585 2804 1789 1122 1208 3291 4465 4195 1396 4652 2001 422 2288 3323 1363 3235 3584 415 2027 5017 759 4432 4134 668
