#!/bin/bash
#SBATCH --job-name=abc_134
#SBATCH --output=logs/abc_134.out
#SBATCH --error=logs/abc_134.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3292 2862 916 1453 395 4038 2344 2893 2074 4421 2542 1593 3756 3758 4193 959 1166 3601 2294 4559 2698 4276 3563 3458 5115 3082
