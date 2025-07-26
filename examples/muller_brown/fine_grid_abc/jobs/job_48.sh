#!/bin/bash
#SBATCH --job-name=abc_48
#SBATCH --output=logs/abc_48.out
#SBATCH --error=logs/abc_48.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 757 169 1560 1746 1774 1732 1761 1572 28 122 722 730 1551 1931 863 61 186 905 1433 1341
