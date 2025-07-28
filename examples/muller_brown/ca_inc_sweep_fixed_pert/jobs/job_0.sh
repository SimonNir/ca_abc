#!/bin/bash
#SBATCH --job-name=abc_0
#SBATCH --output=logs/abc_0.out
#SBATCH --error=logs/abc_0.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 556 412 697 9 558 640 277 46 109
