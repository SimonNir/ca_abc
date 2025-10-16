#!/bin/bash
#SBATCH --job-name=abc_58
#SBATCH --output=logs/abc_58.out
#SBATCH --error=logs/abc_58.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3173 4188 3698 1650 67 4833 1498 4352 3205 3884 3163 171 727 293 2295 3338 2547 190 944 4011 638 2390 2147 3566 4373 654
