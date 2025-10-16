#!/bin/bash
#SBATCH --job-name=abc_88
#SBATCH --output=logs/abc_88.out
#SBATCH --error=logs/abc_88.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 258 1230 770 1064 1745 1627
