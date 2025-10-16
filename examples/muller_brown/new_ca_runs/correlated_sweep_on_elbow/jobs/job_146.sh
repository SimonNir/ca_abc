#!/bin/bash
#SBATCH --job-name=abc_146
#SBATCH --output=logs/abc_146.out
#SBATCH --error=logs/abc_146.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1305 1176 769 711 608 1603 1544 1283 624
