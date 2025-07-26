#!/bin/bash
#SBATCH --job-name=abc_70
#SBATCH --output=logs/abc_70.out
#SBATCH --error=logs/abc_70.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 986 1140 1468 1957 1795 1049 1608 1262 388 138 1272 1483 913 17 529 546 1781 790 1982 1996
