#!/bin/bash
#SBATCH --job-name=abc_42
#SBATCH --output=logs/abc_42.out
#SBATCH --error=logs/abc_42.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1174 1114 508 38 1206 671 253 300 372 320 153 351 951
