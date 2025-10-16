#!/bin/bash
#SBATCH --job-name=abc_22
#SBATCH --output=logs/abc_22.out
#SBATCH --error=logs/abc_22.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2880 2812 2172 4845 2138 2524 4424 387 1113 1615 3499 2473 4139 460 1722 973 2010 301 2077 3006 3394 2616 1936 227 3903 3974
