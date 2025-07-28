#!/bin/bash
#SBATCH --job-name=abc_77
#SBATCH --output=logs/abc_77.out
#SBATCH --error=logs/abc_77.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 533 173 605 341 521 275 769 355 282
