#!/bin/bash
#SBATCH --job-name=abc_81
#SBATCH --output=logs/abc_81.out
#SBATCH --error=logs/abc_81.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1222 1134 906 154 958 705 526 11 348 1295 604 483 1340 1191 1231 1396 1139 1458 762 889
