#!/bin/bash
#SBATCH --job-name=abc_171
#SBATCH --output=logs/abc_171.out
#SBATCH --error=logs/abc_171.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1199 642 639 1275 1538 1358 396 1098 768
