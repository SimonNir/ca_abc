#!/bin/bash
#SBATCH --job-name=abc_14
#SBATCH --output=logs/abc_14.out
#SBATCH --error=logs/abc_14.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 422 507 79 356 1162 938 885 579 453 600 827 947 1205
