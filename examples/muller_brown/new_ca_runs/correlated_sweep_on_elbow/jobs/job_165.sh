#!/bin/bash
#SBATCH --job-name=abc_165
#SBATCH --output=logs/abc_165.out
#SBATCH --error=logs/abc_165.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 832 981 210 684 1280 1105 256 1078 77
