#!/bin/bash
#SBATCH --job-name=abc_143
#SBATCH --output=logs/abc_143.out
#SBATCH --error=logs/abc_143.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4550 2212 5067 1477 2155 801 3942 4720 908 1057 463 3168 4548 2564 2248 2781 3946 3393 369 81 3151 2203 207 3940 3236 504
