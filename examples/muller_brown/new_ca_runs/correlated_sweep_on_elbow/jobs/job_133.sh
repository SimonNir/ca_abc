#!/bin/bash
#SBATCH --job-name=abc_133
#SBATCH --output=logs/abc_133.out
#SBATCH --error=logs/abc_133.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 692 574 730 735 1251 925 1351 952 1005
