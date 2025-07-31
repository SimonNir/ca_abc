#!/bin/bash
#SBATCH --job-name=abc_146
#SBATCH --output=logs/abc_146.out
#SBATCH --error=logs/abc_146.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1836 4616 4433 887 906 3362 2757 3607 3911 3947 4595 684 2088 4271 2578 313 3021 4792 1914 3315 184 3352 790 4389 4793 4523
