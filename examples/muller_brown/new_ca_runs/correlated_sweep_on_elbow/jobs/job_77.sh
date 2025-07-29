#!/bin/bash
#SBATCH --job-name=abc_77
#SBATCH --output=logs/abc_77.out
#SBATCH --error=logs/abc_77.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 402 979 1245 1574 303 1008 231 569 1473
