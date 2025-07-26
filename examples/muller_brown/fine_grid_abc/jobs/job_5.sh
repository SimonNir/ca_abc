#!/bin/bash
#SBATCH --job-name=abc_5
#SBATCH --output=logs/abc_5.out
#SBATCH --error=logs/abc_5.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 503 1861 302 1912 1800 420 1333 1833 1367 1935 887 837 1893 554 237 1849 1063 864 1924 1421
