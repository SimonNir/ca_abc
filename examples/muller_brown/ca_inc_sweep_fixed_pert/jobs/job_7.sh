#!/bin/bash
#SBATCH --job-name=abc_7
#SBATCH --output=logs/abc_7.out
#SBATCH --error=logs/abc_7.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 11 549 393 385 411 211 673 600 669
