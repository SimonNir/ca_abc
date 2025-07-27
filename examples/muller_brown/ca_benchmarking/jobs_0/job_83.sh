#!/bin/bash
#SBATCH --job-name=abc_83
#SBATCH --output=logs/abc_83.out
#SBATCH --error=logs/abc_83.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 614 471 90 936 733 1152 174 186 418 609 103 1190 978
