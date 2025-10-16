#!/bin/bash
#SBATCH --job-name=abc_110
#SBATCH --output=logs/abc_110.out
#SBATCH --error=logs/abc_110.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 121 950 1415 794 163 746 1295 1542 1497
