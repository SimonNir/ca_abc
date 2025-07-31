#!/bin/bash
#SBATCH --job-name=abc_177
#SBATCH --output=logs/abc_177.out
#SBATCH --error=logs/abc_177.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4496 4240 1675 2628 1385 936 2639 5075 3958 2026 4028 76 1670 1631 2563 4270 835 208 2634 2256 35 4502 4381 2264 2188 2225
