#!/bin/bash
#SBATCH --job-name=abc_167
#SBATCH --output=logs/abc_167.out
#SBATCH --error=logs/abc_167.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 988 369 484 1424 1405 1488 1053 227 536
