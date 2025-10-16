#!/bin/bash
#SBATCH --job-name=abc_116
#SBATCH --output=logs/abc_116.out
#SBATCH --error=logs/abc_116.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4428 1094 4713 1403 3404 1026 4096 1203 666 4839 4316 180 377 4196 1377 2472 1646 43 4608 0 9 1805 194 1729 3980 4515
