#!/bin/bash
#SBATCH --job-name=abc_32
#SBATCH --output=logs/abc_32.out
#SBATCH --error=logs/abc_32.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 644 1209 675 1071 328 298 919 179 577 960 44 69 28
