#!/bin/bash
#SBATCH --job-name=abc_2
#SBATCH --output=logs/abc_2.out
#SBATCH --error=logs/abc_2.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 57 767 947 1241 1649 1875 1922 1290 255 55 95 824 869 868 1297 1497 457 69 720 124
