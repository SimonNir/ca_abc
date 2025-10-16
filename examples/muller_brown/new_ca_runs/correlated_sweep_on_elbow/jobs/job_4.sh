#!/bin/bash
#SBATCH --job-name=abc_4
#SBATCH --output=logs/abc_4.out
#SBATCH --error=logs/abc_4.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1244 32 1331 73 442 1432 1590 1402 1145
