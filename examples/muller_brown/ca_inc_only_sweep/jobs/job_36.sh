#!/bin/bash
#SBATCH --job-name=abc_36
#SBATCH --output=logs/abc_36.out
#SBATCH --error=logs/abc_36.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 125 869 905 981 405 534 546 223 355 168
