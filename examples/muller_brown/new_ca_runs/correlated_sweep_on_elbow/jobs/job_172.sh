#!/bin/bash
#SBATCH --job-name=abc_172
#SBATCH --output=logs/abc_172.out
#SBATCH --error=logs/abc_172.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 779 1050 272 450 137 1330 1299 719 315
