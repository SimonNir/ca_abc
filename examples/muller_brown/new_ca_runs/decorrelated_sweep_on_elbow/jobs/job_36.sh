#!/bin/bash
#SBATCH --job-name=abc_36
#SBATCH --output=logs/abc_36.out
#SBATCH --error=logs/abc_36.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2570 3647 1999 1470 2839 3125 4177 3964 2144 880 2602 2529 2039 2098 103 3854 631 1286 2458 4388 4836 322 357 1328 4688 3324
