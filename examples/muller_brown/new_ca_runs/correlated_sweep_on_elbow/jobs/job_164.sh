#!/bin/bash
#SBATCH --job-name=abc_164
#SBATCH --output=logs/abc_164.out
#SBATCH --error=logs/abc_164.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1084 1022 86 117 589 923 1458 281 1316
