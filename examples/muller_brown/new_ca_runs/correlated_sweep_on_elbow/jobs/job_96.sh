#!/bin/bash
#SBATCH --job-name=abc_96
#SBATCH --output=logs/abc_96.out
#SBATCH --error=logs/abc_96.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 291 1366 1041 1017 134 806 1261 1513 1286
