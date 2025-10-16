#!/bin/bash
#SBATCH --job-name=abc_92
#SBATCH --output=logs/abc_92.out
#SBATCH --error=logs/abc_92.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 600 1106 1380 579 1565 1353 504 1339 739
