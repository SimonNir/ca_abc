#!/bin/bash
#SBATCH --job-name=abc_69
#SBATCH --output=logs/abc_69.out
#SBATCH --error=logs/abc_69.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1409 1359 121 1914 1926 764 1789 975 1115 1938 60 1744 1242 1767 967 355 753 412 754 623
