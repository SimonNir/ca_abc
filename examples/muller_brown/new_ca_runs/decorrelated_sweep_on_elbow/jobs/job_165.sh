#!/bin/bash
#SBATCH --job-name=abc_165
#SBATCH --output=logs/abc_165.out
#SBATCH --error=logs/abc_165.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3381 1370 131 4514 3039 1019 4403 2958 4333 2955 3174 2663 2427 2769 2510 3743 3354 2048 453 89 4191 2351 101 3489 929 2694
