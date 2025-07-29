#!/bin/bash
#SBATCH --job-name=abc_65
#SBATCH --output=logs/abc_65.out
#SBATCH --error=logs/abc_65.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 671 339 380 202 285 632 1410 1092 1042
