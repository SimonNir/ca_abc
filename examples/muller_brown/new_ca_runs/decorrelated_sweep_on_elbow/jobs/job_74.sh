#!/bin/bash
#SBATCH --job-name=abc_74
#SBATCH --output=logs/abc_74.out
#SBATCH --error=logs/abc_74.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2025 3467 4704 4847 4189 2419 1524 3929 1551 2514 2186 2132 2133 4411 4103 147 2783 3711 4158 2791 2997 3250 4801 4920 383 140
