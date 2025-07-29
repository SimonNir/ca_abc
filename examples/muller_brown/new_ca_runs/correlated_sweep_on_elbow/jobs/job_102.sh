#!/bin/bash
#SBATCH --job-name=abc_102
#SBATCH --output=logs/abc_102.out
#SBATCH --error=logs/abc_102.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1148 1094 793 1066 1034 454 446 913 391
