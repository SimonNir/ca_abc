#!/bin/bash
#SBATCH --job-name=abc_168
#SBATCH --output=logs/abc_168.out
#SBATCH --error=logs/abc_168.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1026 656 967 747 621 1126 698 1056 650
