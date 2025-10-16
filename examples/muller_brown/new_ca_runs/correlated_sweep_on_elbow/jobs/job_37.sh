#!/bin/bash
#SBATCH --job-name=abc_37
#SBATCH --output=logs/abc_37.out
#SBATCH --error=logs/abc_37.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1459 1611 846 784 850 395 1406 1567 247
