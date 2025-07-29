#!/bin/bash
#SBATCH --job-name=abc_118
#SBATCH --output=logs/abc_118.out
#SBATCH --error=logs/abc_118.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1039 538 361 627 212 242 13 598 47
