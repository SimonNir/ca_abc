#!/bin/bash
#SBATCH --job-name=abc_38
#SBATCH --output=logs/abc_38.out
#SBATCH --error=logs/abc_38.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1783 647 149 1683 1088 1519 114 10 353 300 1284 127 1731 739 1411 441 517 884 1016 1225
