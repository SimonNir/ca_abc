#!/bin/bash
#SBATCH --job-name=abc_4
#SBATCH --output=logs/abc_4.out
#SBATCH --error=logs/abc_4.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 694 175 782 770 807 287 452 462 748
