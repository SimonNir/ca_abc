#!/bin/bash
#SBATCH --job-name=abc_171
#SBATCH --output=logs/abc_171.out
#SBATCH --error=logs/abc_171.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1238 4260 408 2199 4359 2345 2022 4666 2611 4488 3011 2915 4676 1727 3785 4738 1468 3742 393 4604 745 1226 3888 572 938 2425
