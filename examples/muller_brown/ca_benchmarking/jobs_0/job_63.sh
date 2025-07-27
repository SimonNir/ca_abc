#!/bin/bash
#SBATCH --job-name=abc_63
#SBATCH --output=logs/abc_63.out
#SBATCH --error=logs/abc_63.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 394 853 941 937 160 886 93 449 360 1098 637 727 590
