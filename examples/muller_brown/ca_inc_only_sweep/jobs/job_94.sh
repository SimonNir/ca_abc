#!/bin/bash
#SBATCH --job-name=abc_94
#SBATCH --output=logs/abc_94.out
#SBATCH --error=logs/abc_94.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 417 439 527 150 567 365 198 260 264 101
