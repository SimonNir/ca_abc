#!/bin/bash
#SBATCH --job-name=abc_103
#SBATCH --output=logs/abc_103.out
#SBATCH --error=logs/abc_103.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4187 1283 1104 921 4076 1158 747 3715 233 2676 3023 1479 153 2790 4944 382 378 2536 1099 4948 471 3395 2796 475 1364 4888
