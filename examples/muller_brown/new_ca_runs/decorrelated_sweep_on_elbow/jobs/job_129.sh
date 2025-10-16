#!/bin/bash
#SBATCH --job-name=abc_129
#SBATCH --output=logs/abc_129.out
#SBATCH --error=logs/abc_129.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1459 952 4084 447 2900 2934 1635 3723 4697 510 2838 3443 1436 228 2658 2654 1173 1867 4569 4543 731 1147 591 65 5028 2096
