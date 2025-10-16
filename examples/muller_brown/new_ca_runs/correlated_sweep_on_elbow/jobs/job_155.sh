#!/bin/bash
#SBATCH --job-name=abc_155
#SBATCH --output=logs/abc_155.out
#SBATCH --error=logs/abc_155.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 114 63 683 262 1200 799 1335 844 1548
