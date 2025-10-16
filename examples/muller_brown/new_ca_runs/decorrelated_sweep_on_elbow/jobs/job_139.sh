#!/bin/bash
#SBATCH --job-name=abc_139
#SBATCH --output=logs/abc_139.out
#SBATCH --error=logs/abc_139.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2438 1633 1356 495 4037 3511 4911 2335 2150 846 3843 1088 1731 1276 4205 4596 1126 742 695 83 2873 3531 2720 4430 3290 1029
