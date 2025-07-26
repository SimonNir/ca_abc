#!/bin/bash
#SBATCH --job-name=abc_54
#SBATCH --output=logs/abc_54.out
#SBATCH --error=logs/abc_54.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 898 1876 543 323 218 840 1365 849 45 1518 1250 949 948 245 731 1451 999 2 1126 1312
