#!/bin/bash
#SBATCH --job-name=abc_63
#SBATCH --output=logs/abc_63.out
#SBATCH --error=logs/abc_63.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1159 1960 961 596 1346 1653 1645 1281 147 129 397 1326 548 1745 148 1613 797 1522 126 373
