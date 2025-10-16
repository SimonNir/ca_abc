#!/bin/bash
#SBATCH --job-name=abc_71
#SBATCH --output=logs/abc_71.out
#SBATCH --error=logs/abc_71.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2860 345 2923 1589 4326 3010 632 1525 119 2355 3326 3077 2620 4082 720 4945 4525 4831 385 4829 2494 3382 4102 4148 118 3687
