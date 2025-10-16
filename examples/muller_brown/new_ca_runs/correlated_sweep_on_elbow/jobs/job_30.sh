#!/bin/bash
#SBATCH --job-name=abc_30
#SBATCH --output=logs/abc_30.out
#SBATCH --error=logs/abc_30.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1471 394 136 1460 617 1383 330 521 168
