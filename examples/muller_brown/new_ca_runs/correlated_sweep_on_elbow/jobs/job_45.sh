#!/bin/bash
#SBATCH --job-name=abc_45
#SBATCH --output=logs/abc_45.out
#SBATCH --error=logs/abc_45.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 188 1496 857 777 731 17 581 1161 409
