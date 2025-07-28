#!/bin/bash
#SBATCH --job-name=abc_44
#SBATCH --output=logs/abc_44.out
#SBATCH --error=logs/abc_44.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 676 198 117 112 155 170 763 234 283
