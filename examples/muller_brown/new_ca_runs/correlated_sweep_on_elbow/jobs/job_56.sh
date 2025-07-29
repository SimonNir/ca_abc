#!/bin/bash
#SBATCH --job-name=abc_56
#SBATCH --output=logs/abc_56.out
#SBATCH --error=logs/abc_56.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 350 263 1596 564 1268 1012 1254 273 238
