#!/bin/bash
#SBATCH --job-name=abc_67
#SBATCH --output=logs/abc_67.out
#SBATCH --error=logs/abc_67.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1932 1559 851 367 232 1238 1032 465 368 751 874 633 66 1716 743 956 41 1229 721 1347
