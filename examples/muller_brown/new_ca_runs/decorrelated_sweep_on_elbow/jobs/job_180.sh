#!/bin/bash
#SBATCH --job-name=abc_180
#SBATCH --output=logs/abc_180.out
#SBATCH --error=logs/abc_180.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3474 66 2527 3583 2672 2358 712 3959 3625 5090 3493 1656 4294 536 3358 2066 2680 3341 673 3408 1684 2691 1473 3666 4331 920
