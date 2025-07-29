#!/bin/bash
#SBATCH --job-name=abc_174
#SBATCH --output=logs/abc_174.out
#SBATCH --error=logs/abc_174.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1048 211 1430 301 1416 1391 1540 59 1372
