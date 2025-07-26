#!/bin/bash
#SBATCH --job-name=abc_35
#SBATCH --output=logs/abc_35.out
#SBATCH --error=logs/abc_35.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 718 1860 1089 521 1836 1984 164 747 1531 9 1473 1688 1803 1027 1830 263 1934 488 1768 829
