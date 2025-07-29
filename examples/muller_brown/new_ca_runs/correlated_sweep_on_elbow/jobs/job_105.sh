#!/bin/bash
#SBATCH --job-name=abc_105
#SBATCH --output=logs/abc_105.out
#SBATCH --error=logs/abc_105.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1252 1355 934 503 783 697 254 465 1508
