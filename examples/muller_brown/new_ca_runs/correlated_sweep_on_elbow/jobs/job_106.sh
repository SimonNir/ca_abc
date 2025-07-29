#!/bin/bash
#SBATCH --job-name=abc_106
#SBATCH --output=logs/abc_106.out
#SBATCH --error=logs/abc_106.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 751 1503 1143 887 440 812 626 721 1037
