#!/bin/bash
#SBATCH --job-name=abc_74
#SBATCH --output=logs/abc_74.out
#SBATCH --error=logs/abc_74.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 156 1297 1585 744 525 1241 789 1250 602
