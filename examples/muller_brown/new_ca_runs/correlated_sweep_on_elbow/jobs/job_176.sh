#!/bin/bash
#SBATCH --job-name=abc_176
#SBATCH --output=logs/abc_176.out
#SBATCH --error=logs/abc_176.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 640 284 467 1446 1364 1395 1226 963 1515
