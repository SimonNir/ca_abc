#!/bin/bash
#SBATCH --job-name=abc_129
#SBATCH --output=logs/abc_129.out
#SBATCH --error=logs/abc_129.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1267 52 1456 1021 726 39 426 55 1541
