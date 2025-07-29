#!/bin/bash
#SBATCH --job-name=abc_124
#SBATCH --output=logs/abc_124.out
#SBATCH --error=logs/abc_124.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1581 1159 557 1414 314 1419 770 195 1183
