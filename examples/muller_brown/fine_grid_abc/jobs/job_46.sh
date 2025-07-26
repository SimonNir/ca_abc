#!/bin/bash
#SBATCH --job-name=abc_46
#SBATCH --output=logs/abc_46.out
#SBATCH --error=logs/abc_46.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 143 1374 1288 288 737 1034 1400 346 1682 181 516 954 1752 434 1622 606 1479 1884 43 690
