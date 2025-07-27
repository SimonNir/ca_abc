#!/bin/bash
#SBATCH --job-name=abc_58
#SBATCH --output=logs/abc_58.out
#SBATCH --error=logs/abc_58.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 40 954 904 212 906 448 323 756 946 911 76 352 370
