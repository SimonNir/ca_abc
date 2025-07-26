#!/bin/bash
#SBATCH --job-name=abc_53
#SBATCH --output=logs/abc_53.out
#SBATCH --error=logs/abc_53.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 331 1138 1178 1511 750 971 1249 1701 261 1069 1771 1261 1253 1822 1925 1879 1951 1179 735 585
