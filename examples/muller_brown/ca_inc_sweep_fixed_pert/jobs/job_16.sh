#!/bin/bash
#SBATCH --job-name=abc_16
#SBATCH --output=logs/abc_16.out
#SBATCH --error=logs/abc_16.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 370 360 359 743 718 686 309 263 166
