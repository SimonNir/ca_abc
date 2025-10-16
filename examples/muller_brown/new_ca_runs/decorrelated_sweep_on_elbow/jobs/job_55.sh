#!/bin/bash
#SBATCH --job-name=abc_55
#SBATCH --output=logs/abc_55.out
#SBATCH --error=logs/abc_55.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4167 3498 976 4685 4449 2269 1776 343 3308 4971 2850 4638 3297 3776 2846 3923 1877 223 1603 136 3616 2842 4878 617 2107 911
