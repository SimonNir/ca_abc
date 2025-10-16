#!/bin/bash
#SBATCH --job-name=abc_119
#SBATCH --output=logs/abc_119.out
#SBATCH --error=logs/abc_119.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 524 54 431 797 1427 433 1052 1326 712
