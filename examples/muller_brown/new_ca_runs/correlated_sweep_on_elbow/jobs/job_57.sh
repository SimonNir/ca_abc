#!/bin/bash
#SBATCH --job-name=abc_57
#SBATCH --output=logs/abc_57.out
#SBATCH --error=logs/abc_57.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 985 827 140 69 1291 1049 1525 81 1206
