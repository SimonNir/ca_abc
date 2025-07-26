#!/bin/bash
#SBATCH --job-name=abc_61
#SBATCH --output=logs/abc_61.out
#SBATCH --error=logs/abc_61.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1189 634 1228 67 1046 1118 1155 1985 1302 534 80 327 1098 1108 783 1219 220 1848 1165 1627
