#!/bin/bash
#SBATCH --job-name=abc_45
#SBATCH --output=logs/abc_45.out
#SBATCH --error=logs/abc_45.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1541 213 533 157 1059 1815 1420 283 1385 77 972 1687 1695 1317 1676 915 1176 1345 1773 724
