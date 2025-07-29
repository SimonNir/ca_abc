#!/bin/bash
#SBATCH --job-name=abc_14
#SBATCH --output=logs/abc_14.out
#SBATCH --error=logs/abc_14.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1027 1439 1411 1579 1614 292 1168 1015 993
