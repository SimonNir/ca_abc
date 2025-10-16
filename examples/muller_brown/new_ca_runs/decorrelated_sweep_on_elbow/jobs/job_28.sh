#!/bin/bash
#SBATCH --job-name=abc_28
#SBATCH --output=logs/abc_28.out
#SBATCH --error=logs/abc_28.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2359 266 3383 4631 132 3148 1715 2367 3895 3096 58 4165 1637 586 4074 1894 1085 4171 1357 606 4698 289 2148 211 3473 704
