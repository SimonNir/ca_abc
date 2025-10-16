#!/bin/bash
#SBATCH --job-name=abc_130
#SBATCH --output=logs/abc_130.out
#SBATCH --error=logs/abc_130.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 521 534 583 63 763 1586 1186 413 481 1192 2296 1800 1513
