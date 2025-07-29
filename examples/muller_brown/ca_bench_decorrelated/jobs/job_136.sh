#!/bin/bash
#SBATCH --job-name=abc_136
#SBATCH --output=logs/abc_136.out
#SBATCH --error=logs/abc_136.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2289 863 2286 857 2430 1975 1384 1260 768 1919 1625 1420 0
