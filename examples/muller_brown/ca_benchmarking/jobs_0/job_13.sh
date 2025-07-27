#!/bin/bash
#SBATCH --job-name=abc_13
#SBATCH --output=logs/abc_13.out
#SBATCH --error=logs/abc_13.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1138 597 1006 956 782 781 118 486 520 1042 173 361 273
