#!/bin/bash
#SBATCH --job-name=abc_87
#SBATCH --output=logs/abc_87.out
#SBATCH --error=logs/abc_87.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 561 552 243 419 22 705 764 10 1512
