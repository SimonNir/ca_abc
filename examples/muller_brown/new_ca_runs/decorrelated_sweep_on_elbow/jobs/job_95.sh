#!/bin/bash
#SBATCH --job-name=abc_95
#SBATCH --output=logs/abc_95.out
#SBATCH --error=logs/abc_95.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3270 2247 139 2665 4694 37 1112 429 1852 4085 828 2607 2704 69 4901 1395 2004 404 2913 4339 1006 4902 349 252 3465 1041
