#!/bin/bash
#SBATCH --job-name=abc_68
#SBATCH --output=logs/abc_68.out
#SBATCH --error=logs/abc_68.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 547 1313 1128 826 795 416 1598 249 725 1390 1356 13 1639 594 991 295 1210 1662 1328 1100
