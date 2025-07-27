#!/bin/bash
#SBATCH --job-name=abc_0
#SBATCH --output=logs/abc_0.out
#SBATCH --error=logs/abc_0.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 168 902 755 112 555 411 392 340 825 815 723 108 226
