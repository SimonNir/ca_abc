#!/bin/bash
#SBATCH --job-name=abc_134
#SBATCH --output=logs/abc_134.out
#SBATCH --error=logs/abc_134.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1619 2287 1355 2119 1641 2026 1269 2035 1309 1191 2470 493 886
