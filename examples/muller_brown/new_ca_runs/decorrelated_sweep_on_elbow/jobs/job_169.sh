#!/bin/bash
#SBATCH --job-name=abc_169
#SBATCH --output=logs/abc_169.out
#SBATCH --error=logs/abc_169.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4448 1491 2642 2033 580 601 847 3346 215 2426 356 2400 1977 4129 3210 961 2788 4607 2139 4606 702 3965 1847 3048 1966 3963
