#!/bin/bash
#SBATCH --job-name=abc_177
#SBATCH --output=logs/abc_177.out
#SBATCH --error=logs/abc_177.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1468 878 1253 780 933 553 592 1070 1452
