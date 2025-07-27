#!/bin/bash
#SBATCH --job-name=abc_61
#SBATCH --output=logs/abc_61.out
#SBATCH --error=logs/abc_61.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 169 716 1178 661 1145 368 930 524 1036 560 977 865 971
