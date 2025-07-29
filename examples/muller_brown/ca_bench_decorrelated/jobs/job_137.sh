#!/bin/bash
#SBATCH --job-name=abc_137
#SBATCH --output=logs/abc_137.out
#SBATCH --error=logs/abc_137.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1991 1549 1817 2484 2336 2084 1801 122 669 1879 723 2449 457
