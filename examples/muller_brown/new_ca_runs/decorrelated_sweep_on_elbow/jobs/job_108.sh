#!/bin/bash
#SBATCH --job-name=abc_108
#SBATCH --output=logs/abc_108.out
#SBATCH --error=logs/abc_108.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4617 3649 743 4782 3391 4769 4146 179 1098 3611 4199 3835 7 4587 1849 3051 736 3026 3153 2557 1548 152 2914 3691 2187 2185
