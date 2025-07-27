#!/bin/bash
#SBATCH --job-name=abc_68
#SBATCH --output=logs/abc_68.out
#SBATCH --error=logs/abc_68.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 915 1127 747 739 324 57 1130 194 765 861 1087 242 147
