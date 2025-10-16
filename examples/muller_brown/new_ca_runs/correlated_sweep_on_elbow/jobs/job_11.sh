#!/bin/bash
#SBATCH --job-name=abc_11
#SBATCH --output=logs/abc_11.out
#SBATCH --error=logs/abc_11.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 229 738 33 1322 811 1616 261 1555 74
