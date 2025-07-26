#!/bin/bash
#SBATCH --job-name=abc_49
#SBATCH --output=logs/abc_49.out
#SBATCH --error=logs/abc_49.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1429 1965 1174 895 892 1878 911 1162 1403 476 1594 1556 945 907 1750 456 857 1529 703 847
