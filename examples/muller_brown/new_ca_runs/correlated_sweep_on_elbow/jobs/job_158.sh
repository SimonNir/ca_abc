#!/bin/bash
#SBATCH --job-name=abc_158
#SBATCH --output=logs/abc_158.out
#SBATCH --error=logs/abc_158.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 672 269 1293 975 1187 665 1530 882 1453
