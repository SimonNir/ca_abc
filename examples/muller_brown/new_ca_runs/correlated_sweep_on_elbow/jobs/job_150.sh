#!/bin/bash
#SBATCH --job-name=abc_150
#SBATCH --output=logs/abc_150.out
#SBATCH --error=logs/abc_150.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 62 335 805 707 1278 15 895 1132 899
