#!/bin/bash
#SBATCH --job-name=abc_64
#SBATCH --output=logs/abc_64.out
#SBATCH --error=logs/abc_64.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1362 856 893 1302 1020 225 1138 1445 795
