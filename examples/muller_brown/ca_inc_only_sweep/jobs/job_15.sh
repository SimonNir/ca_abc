#!/bin/bash
#SBATCH --job-name=abc_15
#SBATCH --output=logs/abc_15.out
#SBATCH --error=logs/abc_15.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 665 117 992 859 68 499 614 247 280 224
