#!/bin/bash
#SBATCH --job-name=abc_16
#SBATCH --output=logs/abc_16.out
#SBATCH --error=logs/abc_16.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 691 146 1109 337 150 333 1482 1449 1336
