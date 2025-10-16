#!/bin/bash
#SBATCH --job-name=abc_173
#SBATCH --output=logs/abc_173.out
#SBATCH --error=logs/abc_173.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 586 1454 328 713 120 1352 453 1235 1517
