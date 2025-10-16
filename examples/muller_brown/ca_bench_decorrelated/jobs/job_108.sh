#!/bin/bash
#SBATCH --job-name=abc_108
#SBATCH --output=logs/abc_108.out
#SBATCH --error=logs/abc_108.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1960 81 703 450 2186 1608
