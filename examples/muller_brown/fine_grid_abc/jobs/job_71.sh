#!/bin/bash
#SBATCH --job-name=abc_71
#SBATCH --output=logs/abc_71.out
#SBATCH --error=logs/abc_71.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1906 427 1322 1989 152 1234 1018 1147 1742 1068 1417 834 1675 1436 1450 1763 749 1195 1379 658
