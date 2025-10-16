#!/bin/bash
#SBATCH --job-name=abc_54
#SBATCH --output=logs/abc_54.out
#SBATCH --error=logs/abc_54.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 613 4978 1207 3696 2046 4216 2878 2898 2228 4439 1271 4701 3225 3556 1900 1828 257 1458 4784 1801 4207 1604 3480 2238 4993 553
