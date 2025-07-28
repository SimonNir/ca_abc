#!/bin/bash
#SBATCH --job-name=abc_23
#SBATCH --output=logs/abc_23.out
#SBATCH --error=logs/abc_23.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 589 877 964 180 154 780 574 642 908 954
