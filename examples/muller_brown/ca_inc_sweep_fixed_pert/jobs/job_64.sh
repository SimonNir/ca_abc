#!/bin/bash
#SBATCH --job-name=abc_64
#SBATCH --output=logs/abc_64.out
#SBATCH --error=logs/abc_64.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 712 271 226 108 500 409 349 81 620
