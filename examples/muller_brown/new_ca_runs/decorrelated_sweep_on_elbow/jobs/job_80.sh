#!/bin/bash
#SBATCH --job-name=abc_80
#SBATCH --output=logs/abc_80.out
#SBATCH --error=logs/abc_80.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3760 4357 1695 1974 2581 3318 780 2954 3861 458 3094 1344 3310 800 4695 2525 4487 3657 4485 3003 1230 1886 4407 1016 1408 2646
