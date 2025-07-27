#!/bin/bash
#SBATCH --job-name=abc_48
#SBATCH --output=logs/abc_48.out
#SBATCH --error=logs/abc_48.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 847 1026 384 1110 1091 894 1101 434 472 893 185 878 562
