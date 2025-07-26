#!/bin/bash
#SBATCH --job-name=abc_28
#SBATCH --output=logs/abc_28.out
#SBATCH --error=logs/abc_28.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1386 432 885 1964 379 719 402 6 953 24 91 1072 763 776 592 1006 1609 1437 806 343
