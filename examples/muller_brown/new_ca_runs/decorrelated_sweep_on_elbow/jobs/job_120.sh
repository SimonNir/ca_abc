#!/bin/bash
#SBATCH --job-name=abc_120
#SBATCH --output=logs/abc_120.out
#SBATCH --error=logs/abc_120.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2942 53 4567 4556 3860 1315 1933 421 664 3452 2684 2263 2250 4759 3572 3718 2666 291 1842 1178 4186 3795 762 3027 2085 2669
