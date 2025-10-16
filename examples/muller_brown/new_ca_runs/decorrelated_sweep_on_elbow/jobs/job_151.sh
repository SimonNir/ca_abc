#!/bin/bash
#SBATCH --job-name=abc_151
#SBATCH --output=logs/abc_151.out
#SBATCH --error=logs/abc_151.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4921 1683 364 718 556 3693 4647 2496 651 570 4325 2136 2450 4767 3411 1766 2554 3534 5053 784 1183 4692 3305 3455 934 1618
