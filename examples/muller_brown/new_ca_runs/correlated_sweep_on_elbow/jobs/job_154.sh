#!/bin/bash
#SBATCH --job-name=abc_154
#SBATCH --output=logs/abc_154.out
#SBATCH --error=logs/abc_154.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1604 511 1009 252 830 213 1332 1535 246
