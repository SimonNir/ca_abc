#!/bin/bash
#SBATCH --job-name=abc_82
#SBATCH --output=logs/abc_82.out
#SBATCH --error=logs/abc_82.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1170 751 665 383 43 1066 790 955 1057 563 318 1153 664
