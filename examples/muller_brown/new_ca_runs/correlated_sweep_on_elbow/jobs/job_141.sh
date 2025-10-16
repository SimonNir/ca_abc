#!/bin/bash
#SBATCH --job-name=abc_141
#SBATCH --output=logs/abc_141.out
#SBATCH --error=logs/abc_141.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 695 157 918 1501 891 129 803 415 732
