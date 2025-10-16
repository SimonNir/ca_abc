#!/bin/bash
#SBATCH --job-name=abc_170
#SBATCH --output=logs/abc_170.out
#SBATCH --error=logs/abc_170.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3700 3853 4516 4599 4170 4895 3859 3650 5111 4351 1797 5022 1451 63 1973 2895 4043 4635 501 2630 1290 3062 1439 3138 1437 4133
