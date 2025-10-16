#!/bin/bash
#SBATCH --job-name=abc_135
#SBATCH --output=logs/abc_135.out
#SBATCH --error=logs/abc_135.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1374 840 1247 1004 1146 596 1160 1191 1303
