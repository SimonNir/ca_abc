#!/bin/bash
#SBATCH --job-name=abc_3
#SBATCH --output=logs/abc_3.out
#SBATCH --error=logs/abc_3.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 739 1323 2103 1747 657 1454
