#!/bin/bash
#SBATCH --job-name=abc_75
#SBATCH --output=logs/abc_75.out
#SBATCH --error=logs/abc_75.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1595 155 843 324 577 615 193 846 528 1762 952 1040 664 1334 1062 71 286 1005 1136 454
