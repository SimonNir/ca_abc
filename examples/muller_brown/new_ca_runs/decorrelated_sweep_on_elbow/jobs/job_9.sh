#!/bin/bash
#SBATCH --job-name=abc_9
#SBATCH --output=logs/abc_9.out
#SBATCH --error=logs/abc_9.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3337 3279 569 3349 3560 4258 5068 2 177 4658 3417 2018 1448 4732 3247 2373 2572 44 4739 4804 4445 2075 4882 1912 2537 4651
