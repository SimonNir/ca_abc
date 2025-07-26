#!/bin/bash
#SBATCH --job-name=abc_57
#SBATCH --output=logs/abc_57.out
#SBATCH --error=logs/abc_57.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 707 1336 1471 1405 1493 423 1208 352 1320 882 931 90 1672 472 822 631 1661 1865 671 668
