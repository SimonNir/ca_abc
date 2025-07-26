#!/bin/bash
#SBATCH --job-name=abc_40
#SBATCH --output=logs/abc_40.out
#SBATCH --error=logs/abc_40.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1106 1689 1664 492 1080 409 59 1500 378 216 1814 206 741 1928 1777 485 1022 1940 1057 1476
