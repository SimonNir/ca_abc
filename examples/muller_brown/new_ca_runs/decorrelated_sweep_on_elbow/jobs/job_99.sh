#!/bin/bash
#SBATCH --job-name=abc_99
#SBATCH --output=logs/abc_99.out
#SBATCH --error=logs/abc_99.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2605 1980 4998 3786 4799 3788 3986 2439 4868 75 3653 1535 4573 2938 3651 803 3874 3722 3659 335 604 1931 1674 1855 70 148
