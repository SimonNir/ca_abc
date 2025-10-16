#!/bin/bash
#SBATCH --job-name=abc_69
#SBATCH --output=logs/abc_69.out
#SBATCH --error=logs/abc_69.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1224 3463 3539 3243 3159 4597 3133 398 4286 1480 216 2989 2763 1338 4354 4122 3154 3855 3066 4665 3591 1526 715 4466 644 1073
