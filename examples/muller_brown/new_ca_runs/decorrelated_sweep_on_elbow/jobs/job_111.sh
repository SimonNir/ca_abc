#!/bin/bash
#SBATCH --job-name=abc_111
#SBATCH --output=logs/abc_111.out
#SBATCH --error=logs/abc_111.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 518 4542 491 3294 1897 3180 374 4247 609 4163 2765 648 1657 948 1331 3444 1783 1321 5045 4292 5019 2127 2780 2924 4447 4725
