#!/bin/bash
#SBATCH --job-name=abc_163
#SBATCH --output=logs/abc_163.out
#SBATCH --error=logs/abc_163.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 649 1013 28 1315 839 144 980 458 611
