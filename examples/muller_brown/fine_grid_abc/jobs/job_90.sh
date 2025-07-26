#!/bin/bash
#SBATCH --job-name=abc_90
#SBATCH --output=logs/abc_90.out
#SBATCH --error=logs/abc_90.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 934 677 940 438 1166 1324 125 1489 1196 385 582 252 1892 1045 1913 1579 553 1158 1536 537
