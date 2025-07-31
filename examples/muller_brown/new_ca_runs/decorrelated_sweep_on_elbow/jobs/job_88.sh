#!/bin/bash
#SBATCH --job-name=abc_88
#SBATCH --output=logs/abc_88.out
#SBATCH --error=logs/abc_88.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1414 2629 267 1527 1698 1780 1518 1124 4886 1494 1341 1924 4827 1513 198 3416 4991 2756 1334 774 1859 2159 3059 3813 4290 1628
