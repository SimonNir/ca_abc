#!/bin/bash
#SBATCH --job-name=abc_76
#SBATCH --output=logs/abc_76.out
#SBATCH --error=logs/abc_76.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1080 1016 508 563 1582 1369 351 283 1576
