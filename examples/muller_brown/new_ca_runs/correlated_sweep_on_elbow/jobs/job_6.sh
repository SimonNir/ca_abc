#!/bin/bash
#SBATCH --job-name=abc_6
#SBATCH --output=logs/abc_6.out
#SBATCH --error=logs/abc_6.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 30 1412 722 1081 367 1536 604 1111 1552
