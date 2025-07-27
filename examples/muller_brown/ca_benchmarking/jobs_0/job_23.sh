#!/bin/bash
#SBATCH --job-name=abc_23
#SBATCH --output=logs/abc_23.out
#SBATCH --error=logs/abc_23.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 0 999 766 824 588 480 729 107 943 909 880 264 144
