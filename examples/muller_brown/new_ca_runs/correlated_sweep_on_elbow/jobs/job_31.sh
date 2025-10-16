#!/bin/bash
#SBATCH --job-name=abc_31
#SBATCH --output=logs/abc_31.out
#SBATCH --error=logs/abc_31.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 590 1067 1510 234 1195 1393 1075 100 1573
