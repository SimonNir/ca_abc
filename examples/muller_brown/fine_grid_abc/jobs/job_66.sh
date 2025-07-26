#!/bin/bash
#SBATCH --job-name=abc_66
#SBATCH --output=logs/abc_66.out
#SBATCH --error=logs/abc_66.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 980 1127 921 716 1226 1766 696 1137 1946 572 361 215 1590 1747 383 1439 396 1398 171 1443
