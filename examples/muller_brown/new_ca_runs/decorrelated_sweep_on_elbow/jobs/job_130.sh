#!/bin/bash
#SBATCH --job-name=abc_130
#SBATCH --output=logs/abc_130.out
#SBATCH --error=logs/abc_130.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1643 3730 3289 1592 1117 3092 1823 4984 3450 2258 1182 4372 145 4180 874 2825 1014 1386 5035 2405 3837 2451 3217 3586 1399 2151
