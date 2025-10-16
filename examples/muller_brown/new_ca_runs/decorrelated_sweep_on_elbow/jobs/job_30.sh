#!/bin/bash
#SBATCH --job-name=abc_30
#SBATCH --output=logs/abc_30.out
#SBATCH --error=logs/abc_30.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1757 2083 1076 2424 1857 2832 1177 2062 2279 1025 2268 1360 4201 2662 1481 2946 3763 1844 492 900 4341 1919 1251 2325 4718 2149
