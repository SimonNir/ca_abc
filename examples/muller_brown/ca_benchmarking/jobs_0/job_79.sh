#!/bin/bash
#SBATCH --job-name=abc_79
#SBATCH --output=logs/abc_79.out
#SBATCH --error=logs/abc_79.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 51 1167 1085 240 812 929 457 1078 602 338 1199 1202 721
