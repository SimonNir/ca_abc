#!/bin/bash
#SBATCH --job-name=abc_54
#SBATCH --output=logs/abc_54.out
#SBATCH --error=logs/abc_54.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1179 358 518 56 199 934 551 252 656 327 204 130 1039
