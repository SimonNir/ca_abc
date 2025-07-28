#!/bin/bash
#SBATCH --job-name=abc_89
#SBATCH --output=logs/abc_89.out
#SBATCH --error=logs/abc_89.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 168 756 242 163 803 802 235 588 739
