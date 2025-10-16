#!/bin/bash
#SBATCH --job-name=abc_100
#SBATCH --output=logs/abc_100.out
#SBATCH --error=logs/abc_100.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1422 1087 835 1047 938 463 293 1139 1125
