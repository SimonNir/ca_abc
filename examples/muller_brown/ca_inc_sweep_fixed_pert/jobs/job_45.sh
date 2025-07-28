#!/bin/bash
#SBATCH --job-name=abc_45
#SBATCH --output=logs/abc_45.out
#SBATCH --error=logs/abc_45.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 496 472 36 177 310 303 401 460 84
