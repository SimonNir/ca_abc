#!/bin/bash
#SBATCH --job-name=abc_18
#SBATCH --output=logs/abc_18.out
#SBATCH --error=logs/abc_18.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1269 4662 2276 4992 3014 5001 3494 1425 3141 4020 1709 2603 2693 1221 4663 2768 2879 4705 3373 3991 3036 443 1583 3527 2249 4142
