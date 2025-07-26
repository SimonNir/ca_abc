#!/bin/bash
#SBATCH --job-name=abc_19
#SBATCH --output=logs/abc_19.out
#SBATCH --error=logs/abc_19.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1457 1620 519 778 774 1597 639 1054 581 786 871 652 995 518 1293 854 192 850 1827 643
