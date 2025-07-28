#!/bin/bash
#SBATCH --job-name=abc_6
#SBATCH --output=logs/abc_6.out
#SBATCH --error=logs/abc_6.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 308 323 572 25 650 800 185 151 5
