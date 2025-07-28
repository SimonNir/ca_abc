#!/bin/bash
#SBATCH --job-name=abc_3
#SBATCH --output=logs/abc_3.out
#SBATCH --error=logs/abc_3.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 464 107 592 579 293 670 729 683 503
