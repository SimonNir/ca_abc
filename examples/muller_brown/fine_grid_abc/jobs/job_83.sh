#!/bin/bash
#SBATCH --job-name=abc_83
#SBATCH --output=logs/abc_83.out
#SBATCH --error=logs/abc_83.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1274 480 1265 1456 560 210 82 500 899 1381 870 1212 58 1718 1870 474 1255 70 1760 1504
