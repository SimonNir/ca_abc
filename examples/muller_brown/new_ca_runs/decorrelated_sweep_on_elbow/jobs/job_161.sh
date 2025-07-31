#!/bin/bash
#SBATCH --job-name=abc_161
#SBATCH --output=logs/abc_161.out
#SBATCH --error=logs/abc_161.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1596 1512 2687 4943 1507 1907 3554 2160 728 2773 400 2430 5087 2599 2091 1738 3623 2378 4744 4269 2479 2814 2740 2012 1972 3555
