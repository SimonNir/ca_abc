#!/bin/bash
#SBATCH --job-name=abc_55
#SBATCH --output=logs/abc_55.out
#SBATCH --error=logs/abc_55.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 161 82 1000 1077 1347 1472 58 1011 259
