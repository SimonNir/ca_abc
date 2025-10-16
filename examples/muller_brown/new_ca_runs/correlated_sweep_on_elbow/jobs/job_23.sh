#!/bin/bash
#SBATCH --job-name=abc_23
#SBATCH --output=logs/abc_23.out
#SBATCH --error=logs/abc_23.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 667 528 554 488 160 982 362 1340 145
