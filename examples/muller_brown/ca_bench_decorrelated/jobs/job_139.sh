#!/bin/bash
#SBATCH --job-name=abc_139
#SBATCH --output=logs/abc_139.out
#SBATCH --error=logs/abc_139.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 65 165 1397 2179 2264 159 515 2140 724 2433 1590 392 1729
