#!/bin/bash
#SBATCH --job-name=abc_187
#SBATCH --output=logs/abc_187.out
#SBATCH --error=logs/abc_187.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3194 4135 3629 1767 2937 4174 779 4415 2986 4930 4061 1771 50 3215 4130 1316 1090 1532 372 4940 4178 4006 1976 719 283 3610
