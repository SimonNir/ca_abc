#!/bin/bash
#SBATCH --job-name=abc_110
#SBATCH --output=logs/abc_110.out
#SBATCH --error=logs/abc_110.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2178 129 611 3655 3916 1728 957 445 4255 1939 892 1010 3519 3400 2995 1854 2996 4056 4758 3098 3773 450 2214 4066 1545 4753
