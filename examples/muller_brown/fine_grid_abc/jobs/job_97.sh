#!/bin/bash
#SBATCH --job-name=abc_97
#SBATCH --output=logs/abc_97.out
#SBATCH --error=logs/abc_97.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1826 440 1003 511 175 375 1498 504 1648 26 1078 1404 788 1154 939 1380 306 1995 580 1526
