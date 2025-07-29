#!/bin/bash
#SBATCH --job-name=abc_84
#SBATCH --output=logs/abc_84.out
#SBATCH --error=logs/abc_84.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1697 569 1936 913 1542 1182
