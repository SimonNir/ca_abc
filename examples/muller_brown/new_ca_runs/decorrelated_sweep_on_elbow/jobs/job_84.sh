#!/bin/bash
#SBATCH --job-name=abc_84
#SBATCH --output=logs/abc_84.out
#SBATCH --error=logs/abc_84.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 359 4822 469 2067 813 1987 2592 3592 18 4814 5108 339 3407 2953 1469 3540 4871 2036 1927 5041 2246 410 2009 1100 3512 1233
