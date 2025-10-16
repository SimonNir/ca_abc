#!/bin/bash
#SBATCH --job-name=abc_8
#SBATCH --output=logs/abc_8.out
#SBATCH --error=logs/abc_8.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4589 4309 1959 3367 2779 2817 3167 2979 714 3415 3576 3139 4086 4722 3528 361 1718 2440 2515 251 3807 4564 2240 517 133 3070
