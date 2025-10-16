#!/bin/bash
#SBATCH --job-name=abc_107
#SBATCH --output=logs/abc_107.out
#SBATCH --error=logs/abc_107.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 49 1531 760 466 286 492 101 1090 1448
