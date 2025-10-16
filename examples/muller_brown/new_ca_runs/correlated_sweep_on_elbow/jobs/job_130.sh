#!/bin/bash
#SBATCH --job-name=abc_130
#SBATCH --output=logs/abc_130.out
#SBATCH --error=logs/abc_130.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1123 477 1045 763 320 239 447 1379 21
