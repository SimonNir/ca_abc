#!/bin/bash
#SBATCH --job-name=abc_31
#SBATCH --output=logs/abc_31.out
#SBATCH --error=logs/abc_31.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 642 1767 949 2275 860 1501
