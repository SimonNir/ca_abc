#!/bin/bash
#SBATCH --job-name=abc_28
#SBATCH --output=logs/abc_28.out
#SBATCH --error=logs/abc_28.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 890 257 507 778 536 191 157 233 381 873
