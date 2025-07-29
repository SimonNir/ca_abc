#!/bin/bash
#SBATCH --job-name=abc_127
#SBATCH --output=logs/abc_127.out
#SBATCH --error=logs/abc_127.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 165 1420 1292 527 251 928 1096 517 490
