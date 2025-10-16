#!/bin/bash
#SBATCH --job-name=abc_132
#SBATCH --output=logs/abc_132.out
#SBATCH --error=logs/abc_132.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1420 2645 1606 3491 4001 3517 4364 3878 5043 4382 2281 1538 1017 823 3183 144 1775 3161 3143 3103 3961 1963 4737 499 4778 3483
