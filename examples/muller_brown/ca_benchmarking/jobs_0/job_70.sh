#!/bin/bash
#SBATCH --job-name=abc_70
#SBATCH --output=logs/abc_70.out
#SBATCH --error=logs/abc_70.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1193 114 492 750 32 963 65 944 407 990 483 222 939
