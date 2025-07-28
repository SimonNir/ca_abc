#!/bin/bash
#SBATCH --job-name=abc_19
#SBATCH --output=logs/abc_19.out
#SBATCH --error=logs/abc_19.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 453 190 638 444 806 265 475 334 678
