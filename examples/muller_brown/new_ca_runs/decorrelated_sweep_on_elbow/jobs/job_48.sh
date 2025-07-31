#!/bin/bash
#SBATCH --job-name=abc_48
#SBATCH --output=logs/abc_48.out
#SBATCH --error=logs/abc_48.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4366 4707 1895 3104 1162 3798 884 433 4917 623 552 3004 3368 2040 4506 4925 1259 3432 545 993 4641 4379 3132 262 74 1642
