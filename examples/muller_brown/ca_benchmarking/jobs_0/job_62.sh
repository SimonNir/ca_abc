#!/bin/bash
#SBATCH --job-name=abc_62
#SBATCH --output=logs/abc_62.out
#SBATCH --error=logs/abc_62.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1018 1183 8 1165 467 856 1142 127 1201 981 809 301 228
