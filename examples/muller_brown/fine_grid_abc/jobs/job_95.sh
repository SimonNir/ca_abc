#!/bin/bash
#SBATCH --job-name=abc_95
#SBATCH --output=logs/abc_95.out
#SBATCH --error=logs/abc_95.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 649 669 922 354 1342 86 291 197 844 1776 1786 1987 408 51 1806 1101 426 292 655 1233
