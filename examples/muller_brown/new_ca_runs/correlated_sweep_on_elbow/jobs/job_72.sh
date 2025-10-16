#!/bin/bash
#SBATCH --job-name=abc_72
#SBATCH --output=logs/abc_72.out
#SBATCH --error=logs/abc_72.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 992 1486 352 542 1387 1213 469 197 1076
