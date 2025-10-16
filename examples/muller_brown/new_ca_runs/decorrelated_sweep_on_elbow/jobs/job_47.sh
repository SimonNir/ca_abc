#!/bin/bash
#SBATCH --job-name=abc_47
#SBATCH --output=logs/abc_47.out
#SBATCH --error=logs/abc_47.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4073 1375 3298 1440 1948 3079 3372 4884 1134 274 1819 1818 204 3428 1013 4935 141 2944 3886 4475 980 1078 169 1074 1724 705
