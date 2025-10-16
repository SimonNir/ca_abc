#!/bin/bash
#SBATCH --job-name=abc_156
#SBATCH --output=logs/abc_156.out
#SBATCH --error=logs/abc_156.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1314 1285 573 214 1613 1112 1172 1259 19
