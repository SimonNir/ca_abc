#!/bin/bash
#SBATCH --job-name=abc_25
#SBATCH --output=logs/abc_25.out
#SBATCH --error=logs/abc_25.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3643 2692 2903 2413 4072 1916 1682 2594 4825 1872 4248 2456 621 2116 4464 1796 3320 3862 675 138 2713 1051 2011 3366 4773 1265
