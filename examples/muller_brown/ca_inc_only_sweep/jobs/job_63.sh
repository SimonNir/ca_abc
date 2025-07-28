#!/bin/bash
#SBATCH --job-name=abc_63
#SBATCH --output=logs/abc_63.out
#SBATCH --error=logs/abc_63.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 442 934 841 927 889 833 633 380 528 602
