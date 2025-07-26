#!/bin/bash
#SBATCH --job-name=abc_76
#SBATCH --output=logs/abc_76.out
#SBATCH --error=logs/abc_76.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1967 1168 141 284 891 1992 458 226 575 942 1628 287 1651 1799 1366 601 787 358 540 678
