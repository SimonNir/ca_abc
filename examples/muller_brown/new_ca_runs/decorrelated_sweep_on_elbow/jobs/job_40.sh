#!/bin/bash
#SBATCH --job-name=abc_40
#SBATCH --output=logs/abc_40.out
#SBATCH --error=logs/abc_40.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1850 4131 32 3709 1677 539 4950 5112 1411 162 4397 3164 48 225 859 3251 2105 1198 4533 2204 2340 2627 1045 1298 1188 2518
