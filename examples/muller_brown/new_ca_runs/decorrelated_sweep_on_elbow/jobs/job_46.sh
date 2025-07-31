#!/bin/bash
#SBATCH --job-name=abc_46
#SBATCH --output=logs/abc_46.out
#SBATCH --error=logs/abc_46.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3915 854 4332 3729 1764 2544 10 1901 2538 2259 352 2982 441 960 563 4436 278 2785 3188 2655 1282 1609 594 713 2847 165
