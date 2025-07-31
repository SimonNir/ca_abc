#!/bin/bash
#SBATCH --job-name=abc_196
#SBATCH --output=logs/abc_196.out
#SBATCH --error=logs/abc_196.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 191 3462 4356 4206 4637 5000 2632 159 2582 2320 558 4224 1001 2559 2901 3045 2094 1275 1710 3602 2174 2925 2975 1824
