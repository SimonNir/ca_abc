#!/bin/bash
#SBATCH --job-name=abc_37
#SBATCH --output=logs/abc_37.out
#SBATCH --error=logs/abc_37.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1218 1146 2233 2649 2656 3932 635 4579 862 3775 1105 367 1497 2346 3828 263 1060 3533 3557 4848 955 213 2278 4081 2316 105
