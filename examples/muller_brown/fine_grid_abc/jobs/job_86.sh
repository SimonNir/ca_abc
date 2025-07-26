#!/bin/bash
#SBATCH --job-name=abc_86
#SBATCH --output=logs/abc_86.out
#SBATCH --error=logs/abc_86.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1283 1900 1626 996 254 96 613 460 81 421 656 816 63 700 285 896 1680 277 626 1352
