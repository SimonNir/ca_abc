#!/bin/bash
#SBATCH --job-name=abc_86
#SBATCH --output=logs/abc_86.out
#SBATCH --error=logs/abc_86.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1312 1477 758 551 898 1550 601 1169 873
