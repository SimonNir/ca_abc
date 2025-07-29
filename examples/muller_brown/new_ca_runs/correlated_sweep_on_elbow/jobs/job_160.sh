#!/bin/bash
#SBATCH --job-name=abc_160
#SBATCH --output=logs/abc_160.out
#SBATCH --error=logs/abc_160.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1300 356 759 1587 507 1349 834 479 404
