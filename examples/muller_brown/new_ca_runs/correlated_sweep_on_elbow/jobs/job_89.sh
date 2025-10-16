#!/bin/bash
#SBATCH --job-name=abc_89
#SBATCH --output=logs/abc_89.out
#SBATCH --error=logs/abc_89.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 174 95 495 720 790 471 385 393 1182
