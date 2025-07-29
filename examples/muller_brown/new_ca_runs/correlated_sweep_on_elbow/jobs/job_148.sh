#!/bin/bash
#SBATCH --job-name=abc_148
#SBATCH --output=logs/abc_148.out
#SBATCH --error=logs/abc_148.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 87 848 1543 1485 3 264 1474 1065 1212
