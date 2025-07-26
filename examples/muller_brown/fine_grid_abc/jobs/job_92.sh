#!/bin/bash
#SBATCH --job-name=abc_92
#SBATCH --output=logs/abc_92.out
#SBATCH --error=logs/abc_92.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1971 1601 853 140 1273 1008 814 1862 35 919 464 189 85 1461 595 522 391 1463 990 1223
