#!/bin/bash
#SBATCH --job-name=abc_105
#SBATCH --output=logs/abc_105.out
#SBATCH --error=logs/abc_105.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3831 3927 2089 2226 3681 602 4358 4959 3701 685 2561 974 15 1946 4919 432 3954 3770 1552 3160 487 3203 3191 904 4235 261
