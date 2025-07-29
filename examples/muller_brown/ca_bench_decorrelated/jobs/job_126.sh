#!/bin/bash
#SBATCH --job-name=abc_126
#SBATCH --output=logs/abc_126.out
#SBATCH --error=logs/abc_126.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 542 979 713 100 690 649
