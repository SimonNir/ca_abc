#!/bin/bash
#SBATCH --job-name=abc_185
#SBATCH --output=logs/abc_185.out
#SBATCH --error=logs/abc_185.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 396 4828 3025 4824 2406 1445 2748 2275 2213 23 3827 4024 2058 4022 3033 155 3832 4719 214 370 2195 2054 3791 2904 2217 1371
