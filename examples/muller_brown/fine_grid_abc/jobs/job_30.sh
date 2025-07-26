#!/bin/bash
#SBATCH --job-name=abc_30
#SBATCH --output=logs/abc_30.out
#SBATCH --error=logs/abc_30.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1112 455 1109 44 202 1244 672 1474 1765 1581 29 42 99 5 1247 403 558 418 1355 1503
