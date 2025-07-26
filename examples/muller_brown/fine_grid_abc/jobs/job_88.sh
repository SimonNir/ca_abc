#!/bin/bash
#SBATCH --job-name=abc_88
#SBATCH --output=logs/abc_88.out
#SBATCH --error=logs/abc_88.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 744 867 74 813 1130 1540 1104 1665 1299 1637 1004 1829 839 419 1152 1164 1150 340 200 1600
