#!/bin/bash
#SBATCH --job-name=abc_40
#SBATCH --output=logs/abc_40.out
#SBATCH --error=logs/abc_40.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 126 1418 535 1378 509 173 31 841 534
