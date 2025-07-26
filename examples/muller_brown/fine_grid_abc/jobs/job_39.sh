#!/bin/bash
#SBATCH --job-name=abc_39
#SBATCH --output=logs/abc_39.out
#SBATCH --error=logs/abc_39.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1658 1282 638 1908 1410 793 1882 653 992 811 1525 557 630 139 1845 1821 957 345 1636 1956
