#!/bin/bash
#SBATCH --job-name=abc_96
#SBATCH --output=logs/abc_96.out
#SBATCH --error=logs/abc_96.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1749 1621 1395 1427 591 1428 1573 246 36 1286 759 1125 550 433 586 1972 1577 417 1853 1190
