#!/bin/bash
#SBATCH --job-name=abc_78
#SBATCH --output=logs/abc_78.out
#SBATCH --error=logs/abc_78.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 319 606 816 657 116 1592 258 20 311
