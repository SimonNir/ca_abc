#!/bin/bash
#SBATCH --job-name=abc_10
#SBATCH --output=logs/abc_10.out
#SBATCH --error=logs/abc_10.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 796 209 253 713 1576 103 266 828 1095 120 64 994 376 176 1709 1199 667 151 1102 590
