#!/bin/bash
#SBATCH --job-name=abc_2
#SBATCH --output=logs/abc_2.out
#SBATCH --error=logs/abc_2.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 931 1436 858 1222 868 781 109 674 1174
