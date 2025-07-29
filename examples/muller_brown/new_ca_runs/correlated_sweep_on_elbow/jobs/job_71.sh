#!/bin/bash
#SBATCH --job-name=abc_71
#SBATCH --output=logs/abc_71.out
#SBATCH --error=logs/abc_71.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 193 1033 1467 1407 245 1023 676 1060 1478
