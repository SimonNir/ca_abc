#!/bin/bash
#SBATCH --job-name=abc_128
#SBATCH --output=logs/abc_128.out
#SBATCH --error=logs/abc_128.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1091 1224 401 1196 113 870 102 224 678
