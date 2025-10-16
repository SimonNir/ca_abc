#!/bin/bash
#SBATCH --job-name=abc_28
#SBATCH --output=logs/abc_28.out
#SBATCH --error=logs/abc_28.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 125 215 715 752 969 1526 675 373 1010
