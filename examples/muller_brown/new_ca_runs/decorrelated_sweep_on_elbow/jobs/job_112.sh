#!/bin/bash
#SBATCH --job-name=abc_112
#SBATCH --output=logs/abc_112.out
#SBATCH --error=logs/abc_112.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1270 4852 4691 2988 127 4094 3131 2477 4417 3848 4576 4256 2808 496 2519 325 3550 2297 1663 620 4138 1928 1853 2166 5006 3784
