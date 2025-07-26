#!/bin/bash
#SBATCH --job-name=abc_80
#SBATCH --output=logs/abc_80.out
#SBATCH --error=logs/abc_80.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 395 821 1418 936 430 130 303 369 661 974 159 1466 619 247 509 1739 859 1243 691 628
