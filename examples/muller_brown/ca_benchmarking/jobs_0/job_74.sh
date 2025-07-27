#!/bin/bash
#SBATCH --job-name=abc_74
#SBATCH --output=logs/abc_74.out
#SBATCH --error=logs/abc_74.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1084 369 623 651 920 720 905 931 830 1124 1196 1020 1135
