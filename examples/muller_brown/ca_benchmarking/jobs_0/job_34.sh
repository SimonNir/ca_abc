#!/bin/bash
#SBATCH --job-name=abc_34
#SBATCH --output=logs/abc_34.out
#SBATCH --error=logs/abc_34.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 888 599 180 732 1197 496 192 1122 278 1032 17 641 949
