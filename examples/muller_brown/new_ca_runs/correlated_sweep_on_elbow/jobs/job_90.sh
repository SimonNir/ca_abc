#!/bin/bash
#SBATCH --job-name=abc_90
#SBATCH --output=logs/abc_90.out
#SBATCH --error=logs/abc_90.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 491 1019 223 1158 653 1518 1342 537 1097
