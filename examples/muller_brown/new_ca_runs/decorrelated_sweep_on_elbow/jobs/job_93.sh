#!/bin/bash
#SBATCH --job-name=abc_93
#SBATCH --output=logs/abc_93.out
#SBATCH --error=logs/abc_93.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3661 1875 1287 1110 477 3956 425 531 3905 3189 2461 3112 550 681 1192 1803 2830 4554 4324 4654 1130 3834 3065 5004 71 3850
