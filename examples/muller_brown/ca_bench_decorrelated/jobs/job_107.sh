#!/bin/bash
#SBATCH --job-name=abc_107
#SBATCH --output=logs/abc_107.out
#SBATCH --error=logs/abc_107.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2090 1733 280 888 407 1568
