#!/bin/bash
#SBATCH --job-name=abc_75
#SBATCH --output=logs/abc_75.out
#SBATCH --error=logs/abc_75.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 545 413 1076 142 81 227 687 125 898 1010 71 1065 45
