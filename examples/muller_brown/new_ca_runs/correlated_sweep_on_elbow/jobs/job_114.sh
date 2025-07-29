#!/bin/bash
#SBATCH --job-name=abc_114
#SBATCH --output=logs/abc_114.out
#SBATCH --error=logs/abc_114.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1153 1114 965 1569 1025 914 1457 1248 565
