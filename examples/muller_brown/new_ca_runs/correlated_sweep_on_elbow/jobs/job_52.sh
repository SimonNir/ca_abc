#!/bin/bash
#SBATCH --job-name=abc_52
#SBATCH --output=logs/abc_52.out
#SBATCH --error=logs/abc_52.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 45 185 327 1133 1386 1288 818 583 693
