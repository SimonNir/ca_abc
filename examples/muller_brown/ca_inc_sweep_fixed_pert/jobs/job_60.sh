#!/bin/bash
#SBATCH --job-name=abc_60
#SBATCH --output=logs/abc_60.out
#SBATCH --error=logs/abc_60.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 27 692 188 217 805 41 320 707 759
