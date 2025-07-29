#!/bin/bash
#SBATCH --job-name=abc_59
#SBATCH --output=logs/abc_59.out
#SBATCH --error=logs/abc_59.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 648 927 1776 1744 1337 2138
