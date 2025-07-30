#!/bin/bash
#SBATCH --job-name=abc_38_2
#SBATCH --output=logs/abc_38_2.out
#SBATCH --error=logs/abc_38_2.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 173 50 181 199 184
