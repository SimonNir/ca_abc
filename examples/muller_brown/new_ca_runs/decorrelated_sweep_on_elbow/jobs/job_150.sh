#!/bin/bash
#SBATCH --job-name=abc_150
#SBATCH --output=logs/abc_150.out
#SBATCH --error=logs/abc_150.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1810 5014 4699 2848 3789 1798 1546 2964 2045 351 703 3328 1388 4164 1947 1811 2386 1988 3672 4924 2343 3856 259 2987 554 3891
