#!/bin/bash
#SBATCH --job-name=abc_4
#SBATCH --output=logs/abc_4.out
#SBATCH --error=logs/abc_4.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1106 4350 2099 4861 809 205 4124 1492 2566 4661 3384 1170 4768 4591 3901 2523 1501 1645 4377 4619 3440 3095 2290 2375 3234 1137
