#!/bin/bash
#SBATCH --job-name=abc_35
#SBATCH --output=logs/abc_35.out
#SBATCH --error=logs/abc_35.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 612 1086 216 1172 224 512 969 774 1204 365 1163 799 852
