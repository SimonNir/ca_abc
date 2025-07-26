#!/bin/bash
#SBATCH --job-name=abc_4
#SBATCH --output=logs/abc_4.out
#SBATCH --error=logs/abc_4.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1276 405 437 1748 559 998 1075 789 833 1569 239 977 406 1182 929 1028 909 34 134 1643
