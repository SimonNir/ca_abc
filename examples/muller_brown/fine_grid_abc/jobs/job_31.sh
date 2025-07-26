#!/bin/bash
#SBATCH --job-name=abc_31
#SBATCH --output=logs/abc_31.out
#SBATCH --error=logs/abc_31.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 449 1007 1805 1587 660 1001 1710 1097 1644 1319 1545 229 782 425 597 1043 224 1623 775 1692
