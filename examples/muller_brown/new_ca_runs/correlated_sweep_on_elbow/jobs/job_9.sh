#!/bin/bash
#SBATCH --job-name=abc_9
#SBATCH --output=logs/abc_9.out
#SBATCH --error=logs/abc_9.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 345 1290 199 515 802 1470 1110 135 182
