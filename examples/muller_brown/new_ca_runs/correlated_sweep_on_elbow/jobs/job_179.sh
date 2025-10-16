#!/bin/bash
#SBATCH --job-name=abc_179
#SBATCH --output=logs/abc_179.out
#SBATCH --error=logs/abc_179.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 329 1327 169 164 708 1317 1589 204 308
