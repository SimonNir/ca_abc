#!/bin/bash
#SBATCH --job-name=abc_20
#SBATCH --output=logs/abc_20.out
#SBATCH --error=logs/abc_20.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 973 1285 1011 780 1103 761 1076 1460 1793 201 1894 1481 1566 745 831 1993 1753 298 1412 570
