#!/bin/bash
#SBATCH --job-name=abc_190
#SBATCH --output=logs/abc_190.out
#SBATCH --error=logs/abc_190.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 831 5002 3012 1955 4712 2389 1467 3796 2640 4035 865 1686 4961 2460 3412 901 3258 2337 493 2577 4059 1132 2874 4686 342 1851
