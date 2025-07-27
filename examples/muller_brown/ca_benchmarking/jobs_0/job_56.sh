#!/bin/bash
#SBATCH --job-name=abc_56
#SBATCH --output=logs/abc_56.out
#SBATCH --error=logs/abc_56.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1028 101 268 924 1052 541 26 258 1207 884 1001 1106 743
