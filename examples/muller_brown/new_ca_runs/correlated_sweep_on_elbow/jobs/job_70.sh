#!/bin/bash
#SBATCH --job-name=abc_70
#SBATCH --output=logs/abc_70.out
#SBATCH --error=logs/abc_70.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1208 543 1237 481 1192 939 901 1500 106
