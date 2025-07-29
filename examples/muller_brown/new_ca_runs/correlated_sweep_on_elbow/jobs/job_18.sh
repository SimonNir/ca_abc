#!/bin/bash
#SBATCH --job-name=abc_18
#SBATCH --output=logs/abc_18.out
#SBATCH --error=logs/abc_18.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 568 587 1102 1617 736 377 240 1135 778
