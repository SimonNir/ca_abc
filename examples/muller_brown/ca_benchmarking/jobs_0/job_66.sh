#!/bin/bash
#SBATCH --job-name=abc_66
#SBATCH --output=logs/abc_66.out
#SBATCH --error=logs/abc_66.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 968 219 211 611 172 970 989 96 1053 935 1194 143 166
