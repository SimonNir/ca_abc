#!/bin/bash
#SBATCH --job-name=abc_161
#SBATCH --output=logs/abc_161.out
#SBATCH --error=logs/abc_161.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1313 922 1417 907 1093 1570 354 541 658
