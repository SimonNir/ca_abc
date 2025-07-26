#!/bin/bash
#SBATCH --job-name=abc_98
#SBATCH --output=logs/abc_98.out
#SBATCH --error=logs/abc_98.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1048 578 926 1309 1817 1331 1129 694 1552 422 377 304 1973 1086 174 800 1756 1304 636 617
