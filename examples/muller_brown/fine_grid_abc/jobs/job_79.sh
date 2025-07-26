#!/bin/bash
#SBATCH --job-name=abc_79
#SBATCH --output=logs/abc_79.out
#SBATCH --error=logs/abc_79.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 736 684 1605 1851 1886 1686 1399 371 1494 1950 1936 1990 117 539 710 1901 801 1362 541 1248
