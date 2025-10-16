#!/bin/bash
#SBATCH --job-name=abc_121
#SBATCH --output=logs/abc_121.out
#SBATCH --error=logs/abc_121.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3695 1570 2744 564 5113 3821 3761 114 302 3593 1744 5042 3176 2237 1536 3978 1848 1748 3295 1406 3054 1069 2109 2478 1888 2509
