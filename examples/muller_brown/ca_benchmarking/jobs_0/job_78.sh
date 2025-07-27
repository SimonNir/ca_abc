#!/bin/bash
#SBATCH --job-name=abc_78
#SBATCH --output=logs/abc_78.out
#SBATCH --error=logs/abc_78.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 498 42 1143 1120 22 1046 1022 1175 730 532 21 178 10
