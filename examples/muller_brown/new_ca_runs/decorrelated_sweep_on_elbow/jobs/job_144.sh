#!/bin/bash
#SBATCH --job-name=abc_144
#SBATCH --output=logs/abc_144.out
#SBATCH --error=logs/abc_144.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 555 1050 2079 4563 1080 2921 1052 999 1237 2292 1984 4877 3626 4490 231 2947 3123 31 4115 761 717 513 3839 1612 199 1272
