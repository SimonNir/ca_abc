#!/bin/bash
#SBATCH --job-name=abc_60
#SBATCH --output=logs/abc_60.out
#SBATCH --error=logs/abc_60.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2541 1044 4923 1389 2905 2550 1284 3245 692 1107 3211 2840 1243 2110 2119 3714 889 4873 789 4803 2193 4818 1476 2505 1712 1699
