#!/bin/bash
#SBATCH --job-name=abc_12
#SBATCH --output=logs/abc_12.out
#SBATCH --error=logs/abc_12.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 76 190 1107 624 356 204 1047 486 1758 462 542 364 567 571 650 758 496 1515 294 1635
