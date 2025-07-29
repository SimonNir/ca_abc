#!/bin/bash
#SBATCH --job-name=abc_65
#SBATCH --output=logs/abc_65.out
#SBATCH --error=logs/abc_65.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1194 1726 1885 2152 910 696
