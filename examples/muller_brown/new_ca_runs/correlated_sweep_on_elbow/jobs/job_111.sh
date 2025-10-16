#!/bin/bash
#SBATCH --job-name=abc_111
#SBATCH --output=logs/abc_111.out
#SBATCH --error=logs/abc_111.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 717 222 119 1504 761 629 971 230 418
