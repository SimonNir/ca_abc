#!/bin/bash
#SBATCH --job-name=abc_25
#SBATCH --output=logs/abc_25.out
#SBATCH --error=logs/abc_25.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1073 962 179 1557 192 571 755 302 1155
