#!/bin/bash
#SBATCH --job-name=abc_17
#SBATCH --output=logs/abc_17.out
#SBATCH --error=logs/abc_17.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 388 1363 1554 1520 757 622 151 936 888
