#!/bin/bash
#SBATCH --job-name=abc_138
#SBATCH --output=logs/abc_138.out
#SBATCH --error=logs/abc_138.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 877 249 664 1652 1075 1833 2185 626 98 905 976 36 2106
