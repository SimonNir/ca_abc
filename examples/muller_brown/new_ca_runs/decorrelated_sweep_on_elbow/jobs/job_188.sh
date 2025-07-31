#!/bin/bash
#SBATCH --job-name=abc_188
#SBATCH --output=logs/abc_188.out
#SBATCH --error=logs/abc_188.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4512 3840 3490 2984 4018 877 3819 2260 4897 1580 3063 4317 3116 3140 1601 3172 176 1297 1253 922 1782 520 4386 386 1749 4850
