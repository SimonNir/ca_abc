#!/bin/bash
#SBATCH --job-name=abc_22
#SBATCH --output=logs/abc_22.out
#SBATCH --error=logs/abc_22.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1490 469 1279 293 336 1029 1631 1438 1801 686 1953 183 1170 52 1832 1280 616 1024 888 1673
