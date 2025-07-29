#!/bin/bash
#SBATCH --job-name=abc_138
#SBATCH --output=logs/abc_138.out
#SBATCH --error=logs/abc_138.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 237 688 155 1129 995 14 406 1392 1239
