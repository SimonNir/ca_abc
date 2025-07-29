#!/bin/bash
#SBATCH --job-name=abc_93
#SBATCH --output=logs/abc_93.out
#SBATCH --error=logs/abc_93.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 673 529 24 464 745 411 1428 506 889
