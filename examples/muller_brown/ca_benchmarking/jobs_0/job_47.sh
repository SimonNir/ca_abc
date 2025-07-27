#!/bin/bash
#SBATCH --job-name=abc_47
#SBATCH --output=logs/abc_47.out
#SBATCH --error=logs/abc_47.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 355 427 843 196 813 385 1097 694 877 1109 1072 1005 561
