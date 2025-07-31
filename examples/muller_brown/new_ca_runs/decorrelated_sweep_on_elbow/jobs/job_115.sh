#!/bin/bash
#SBATCH --job-name=abc_115
#SBATCH --output=logs/abc_115.out
#SBATCH --error=logs/abc_115.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1543 3364 4521 3600 4809 3157 529 2224 768 861 3537 2398 4337 1557 2037 4904 4755 157 486 36 3024 910 1790 428 1925 1887
