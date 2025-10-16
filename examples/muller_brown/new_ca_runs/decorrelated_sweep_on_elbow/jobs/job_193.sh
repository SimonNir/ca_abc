#!/bin/bash
#SBATCH --job-name=abc_193
#SBATCH --output=logs/abc_193.out
#SBATCH --error=logs/abc_193.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2976 3518 2533 1463 753 988 4505 2641 4343 2836 320 1008 3797 5027 2892 497 1428 3007 2806 3353 2778 2813 4443 2481 3817 4746
