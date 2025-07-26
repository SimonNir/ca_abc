#!/bin/bash
#SBATCH --job-name=abc_15
#SBATCH --output=logs/abc_15.out
#SBATCH --error=logs/abc_15.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 966 1975 1917 535 1690 56 79 1002 1393 1053 84 132 803 513 1677 848 150 598 698 1464
