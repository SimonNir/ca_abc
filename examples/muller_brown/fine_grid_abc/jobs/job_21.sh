#!/bin/bash
#SBATCH --job-name=abc_21
#SBATCH --output=logs/abc_21.out
#SBATCH --error=logs/abc_21.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 372 173 1495 1416 914 923 706 894 1988 1505 489 118 523 1854 1206 555 625 766 109 1717
