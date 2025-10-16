#!/bin/bash
#SBATCH --job-name=abc_12
#SBATCH --output=logs/abc_12.out
#SBATCH --error=logs/abc_12.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 349 1529 1600 1198 1301 1201 994 1218 771
