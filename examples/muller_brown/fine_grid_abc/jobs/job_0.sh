#!/bin/bash
#SBATCH --job-name=abc_0
#SBATCH --output=logs/abc_0.out
#SBATCH --error=logs/abc_0.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 142 410 1674 208 717 1397 461 1891 1696 448 258 1094 1306 1578 1588 1440 1173 238 1146 1121
