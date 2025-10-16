#!/bin/bash
#SBATCH --job-name=abc_101
#SBATCH --output=logs/abc_101.out
#SBATCH --error=logs/abc_101.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 29 1044 1401 1338 25 531 1258 1284 680
