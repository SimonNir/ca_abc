#!/bin/bash
#SBATCH --job-name=abc_51
#SBATCH --output=logs/abc_51.out
#SBATCH --error=logs/abc_51.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 941 443 935 88 1486 453 702 1723 1902 520 666 1227 621 1264 662 1114 1544 1754 1574 1
