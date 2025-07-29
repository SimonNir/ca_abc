#!/bin/bash
#SBATCH --job-name=abc_96
#SBATCH --output=logs/abc_96.out
#SBATCH --error=logs/abc_96.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1682 1725 1222 1571 2183 2319
