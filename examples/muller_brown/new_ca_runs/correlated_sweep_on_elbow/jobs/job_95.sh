#!/bin/bash
#SBATCH --job-name=abc_95
#SBATCH --output=logs/abc_95.out
#SBATCH --error=logs/abc_95.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 585 201 833 5 929 207 1431 1400 817
