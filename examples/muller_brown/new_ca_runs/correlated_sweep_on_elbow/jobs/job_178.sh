#!/bin/bash
#SBATCH --job-name=abc_178
#SBATCH --output=logs/abc_178.out
#SBATCH --error=logs/abc_178.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 473 383 681 1450 1180 1068 742 631 613
