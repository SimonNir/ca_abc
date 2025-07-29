#!/bin/bash
#SBATCH --job-name=abc_98
#SBATCH --output=logs/abc_98.out
#SBATCH --error=logs/abc_98.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 937 200 859 318 1234 147 664 159 549
