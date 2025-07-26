#!/bin/bash
#SBATCH --job-name=abc_50
#SBATCH --output=logs/abc_50.out
#SBATCH --error=logs/abc_50.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1896 587 1496 305 640 1843 1296 314 1213 1491 1640 72 1703 1391 552 1911 499 804 1026 732
