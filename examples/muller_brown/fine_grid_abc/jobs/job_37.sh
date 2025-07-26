#!/bin/bash
#SBATCH --job-name=abc_37
#SBATCH --output=logs/abc_37.out
#SBATCH --error=logs/abc_37.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 235 685 858 1096 359 153 444 1093 1555 728 1802 1726 230 1298 781 1358 1983 1812 512 1840
