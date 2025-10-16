#!/bin/bash
#SBATCH --job-name=abc_109
#SBATCH --output=logs/abc_109.out
#SBATCH --error=logs/abc_109.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1564 1509 498 821 1421 266 1533 1205 1511
