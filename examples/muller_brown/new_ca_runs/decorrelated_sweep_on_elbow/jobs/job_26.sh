#!/bin/bash
#SBATCH --job-name=abc_26
#SBATCH --output=logs/abc_26.out
#SBATCH --error=logs/abc_26.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2486 1061 5048 4561 305 983 4846 4602 502 630 844 5101 4422 247 1651 3609 4794 1285 1942 2749 4010 4581 2786 84 1667 2056
