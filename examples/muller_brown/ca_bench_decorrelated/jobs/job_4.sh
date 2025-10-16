#!/bin/bash
#SBATCH --job-name=abc_4
#SBATCH --output=logs/abc_4.out
#SBATCH --error=logs/abc_4.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 448 483 1594 2398 1131 1742
