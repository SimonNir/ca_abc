#!/bin/bash
#SBATCH --job-name=abc_147
#SBATCH --output=logs/abc_147.out
#SBATCH --error=logs/abc_147.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 203 659 703 1270 1481 110 184 1455 607
