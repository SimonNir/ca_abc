#!/bin/bash
#SBATCH --job-name=abc_117
#SBATCH --output=logs/abc_117.out
#SBATCH --error=logs/abc_117.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4996 2933 4089 2029 4198 97 3389 2869 414 2466 3919 1197 3369 6 4274 4972 4510 2381 4864 161 4611 2031 3530 4610 1578 1892
