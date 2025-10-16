#!/bin/bash
#SBATCH --job-name=abc_94
#SBATCH --output=logs/abc_94.out
#SBATCH --error=logs/abc_94.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3912 2965 2532 4947 96 1739 2590 4776 4032 2125 2418 4520 1770 3423 4900 1647 1614 5106 1387 4735 1140 2447 1082 1471 3145 13
