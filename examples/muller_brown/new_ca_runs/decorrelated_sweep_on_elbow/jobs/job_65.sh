#!/bin/bash
#SBATCH --job-name=abc_65
#SBATCH --output=logs/abc_65.out
#SBATCH --error=logs/abc_65.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3982 3136 3074 2897 3269 4299 4990 4702 4183 20 3200 4979 4939 1348 5095 4348 2086 4175 870 1261 3030 966 4749 285 2819 4664
