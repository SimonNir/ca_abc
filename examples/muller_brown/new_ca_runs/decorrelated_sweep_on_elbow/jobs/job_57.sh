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

python run_one.py 1794 1827 240 3122 2431 3628 4423 5076 4750 1410 3081 4503 3799 2167 2024 5020 4711 4361 2962 2323 14 1142 3553 1774 1676 2660
