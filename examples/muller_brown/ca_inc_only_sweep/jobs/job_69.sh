#!/bin/bash
#SBATCH --job-name=abc_69
#SBATCH --output=logs/abc_69.out
#SBATCH --error=logs/abc_69.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 409 29 977 37 277 697 679 872 299 51
