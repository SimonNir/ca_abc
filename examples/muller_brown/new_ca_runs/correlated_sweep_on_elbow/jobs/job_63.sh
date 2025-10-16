#!/bin/bash
#SBATCH --job-name=abc_63
#SBATCH --output=logs/abc_63.out
#SBATCH --error=logs/abc_63.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 558 241 11 539 774 414 1373 513 915
