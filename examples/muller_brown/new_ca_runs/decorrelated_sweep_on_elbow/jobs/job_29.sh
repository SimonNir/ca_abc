#!/bin/bash
#SBATCH --job-name=abc_29
#SBATCH --output=logs/abc_29.out
#SBATCH --error=logs/abc_29.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4000 4334 782 1235 2920 2633 2637 1005 646 2234 519 1485 1993 5107 3521 4418 3047 4771 4042 5118 1189 4279 2584 312 3679 2055
