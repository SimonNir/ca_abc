#!/bin/bash
#SBATCH --job-name=abc_24
#SBATCH --output=logs/abc_24.out
#SBATCH --error=logs/abc_24.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 401 929 920 704 87 565 854 649 336 307
