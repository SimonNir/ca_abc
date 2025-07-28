#!/bin/bash
#SBATCH --job-name=abc_79
#SBATCH --output=logs/abc_79.out
#SBATCH --error=logs/abc_79.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 587 538 189 129 367 224 347 765 699
