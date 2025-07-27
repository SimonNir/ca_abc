#!/bin/bash
#SBATCH --job-name=abc_9
#SBATCH --output=logs/abc_9.out
#SBATCH --error=logs/abc_9.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 170 293 544 429 746 260 652 1041 476 290 792 316 918
