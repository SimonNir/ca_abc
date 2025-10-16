#!/bin/bash
#SBATCH --job-name=abc_89
#SBATCH --output=logs/abc_89.out
#SBATCH --error=logs/abc_89.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2725 3542 3707 3953 1343 568 1600 3988 533 2215 3746 2626 2752 1691 3186 1087 3459 1475 1219 4458 3017 811 3549 186 871 399
