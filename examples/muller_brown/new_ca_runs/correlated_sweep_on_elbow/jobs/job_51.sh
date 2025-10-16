#!/bin/bash
#SBATCH --job-name=abc_51
#SBATCH --output=logs/abc_51.out
#SBATCH --error=logs/abc_51.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 323 172 861 1281 217 347 131 920 1265
