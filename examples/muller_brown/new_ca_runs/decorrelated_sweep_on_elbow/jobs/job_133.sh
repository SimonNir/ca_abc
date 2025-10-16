#!/bin/bash
#SBATCH --job-name=abc_133
#SBATCH --output=logs/abc_133.out
#SBATCH --error=logs/abc_133.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 5064 334 3617 972 595 3056 1533 3370 657 1679 3604 154 3190 3989 543 4016 2499 4621 1602 3803 54 3740 3524 1662 3892 353
