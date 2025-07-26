#!/bin/bash
#SBATCH --job-name=abc_77
#SBATCH --output=logs/abc_77.out
#SBATCH --error=logs/abc_77.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1818 318 1809 689 1151 1736 1751 1188 687 646 299 270 1092 693 1384 1181 1351 574 1721 918
