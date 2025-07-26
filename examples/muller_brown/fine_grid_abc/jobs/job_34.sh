#!/bin/bash
#SBATCH --job-name=abc_34
#SBATCH --output=logs/abc_34.out
#SBATCH --error=logs/abc_34.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 332 611 1521 805 588 1124 1270 21 178 1667 573 1120 1172 1105 1311 841 699 1869 214 1023
