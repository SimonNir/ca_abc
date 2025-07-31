#!/bin/bash
#SBATCH --job-name=abc_175
#SBATCH --output=logs/abc_175.out
#SBATCH --error=logs/abc_175.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2782 4127 2034 2245 1560 2236 4497 2983 676 2615 4480 2528 3100 5078 4670 3792 1244 1368 3574 709 998 4047 2042 3152 3985 1781
