#!/bin/bash
#SBATCH --job-name=abc_94
#SBATCH --output=logs/abc_94.out
#SBATCH --error=logs/abc_94.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 603 823 1388 555 1308 588 379 1444 295
