#!/bin/bash
#SBATCH --job-name=abc_129
#SBATCH --output=logs/abc_129.out
#SBATCH --error=logs/abc_129.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1473 1285 2115 539 1540 320 1037 1020 1996 1072 1098 2353 1885
