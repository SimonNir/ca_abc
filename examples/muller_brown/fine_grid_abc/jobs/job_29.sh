#!/bin/bash
#SBATCH --job-name=abc_29
#SBATCH --output=logs/abc_29.out
#SBATCH --error=logs/abc_29.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 264 902 1432 1863 1349 1825 1700 873 241 1268 1506 1570 879 1959 1714 1585 329 1513 308 429
