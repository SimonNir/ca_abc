#!/bin/bash
#SBATCH --job-name=abc_3
#SBATCH --output=logs/abc_3.out
#SBATCH --error=logs/abc_3.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 605 2531 4657 1391 246 4517 2332 3973 4593 1023 1421 2882 2707 2844 350 2334 1561 3388 3849 4933 1922 4867 1540 2352 1302 2945
