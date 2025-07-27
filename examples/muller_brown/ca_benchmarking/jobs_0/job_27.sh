#!/bin/bash
#SBATCH --job-name=abc_27
#SBATCH --output=logs/abc_27.out
#SBATCH --error=logs/abc_27.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 99 817 159 1014 1100 659 849 1133 281 440 105 643 794
