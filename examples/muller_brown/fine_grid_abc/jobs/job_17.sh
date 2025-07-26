#!/bin/bash
#SBATCH --job-name=abc_17
#SBATCH --output=logs/abc_17.out
#SBATCH --error=logs/abc_17.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 350 1149 714 349 1547 1375 1019 1143 1969 701 1185 568 495 970 602 1183 644 484 1719 333
