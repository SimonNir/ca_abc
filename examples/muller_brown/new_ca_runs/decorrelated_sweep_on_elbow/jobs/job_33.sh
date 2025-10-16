#!/bin/bash
#SBATCH --job-name=abc_33
#SBATCH --output=logs/abc_33.out
#SBATCH --error=logs/abc_33.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 5023 2322 3858 3185 3119 3955 4928 4600 1982 3547 2731 3118 4800 2287 22 4374 3402 3889 2421 468 4117 1393 202 4041 4966 2827
