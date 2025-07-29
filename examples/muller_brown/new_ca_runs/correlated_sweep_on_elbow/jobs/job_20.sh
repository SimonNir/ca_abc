#!/bin/bash
#SBATCH --job-name=abc_20
#SBATCH --output=logs/abc_20.out
#SBATCH --error=logs/abc_20.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1442 358 332 299 1435 366 1491 1337 1029
