#!/bin/bash
#SBATCH --job-name=abc_22
#SBATCH --output=logs/abc_22.out
#SBATCH --error=logs/abc_22.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 495 361 188 785 738 531 295 734 791 868
