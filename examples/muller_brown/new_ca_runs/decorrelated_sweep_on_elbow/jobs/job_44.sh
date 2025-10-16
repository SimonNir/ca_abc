#!/bin/bash
#SBATCH --job-name=abc_44
#SBATCH --output=logs/abc_44.out
#SBATCH --error=logs/abc_44.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3192 3487 1204 3162 4226 4929 178 4632 1503 4039 2513 1701 500 4296 2350 4869 4585 108 4646 2142 2169 3427 1413 2364 2810 3300
