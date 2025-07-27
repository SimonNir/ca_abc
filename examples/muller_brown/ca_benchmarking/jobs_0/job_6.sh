#!/bin/bash
#SBATCH --job-name=abc_6
#SBATCH --output=logs/abc_6.out
#SBATCH --error=logs/abc_6.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 46 550 1137 238 575 1198 952 1131 481 538 839 704 632
