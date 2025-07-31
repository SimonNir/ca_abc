#!/bin/bash
#SBATCH --job-name=abc_100
#SBATCH --output=logs/abc_100.out
#SBATCH --error=logs/abc_100.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4450 163 3009 4210 1832 4568 1434 1565 1325 3313 2621 2504 2262 1279 2608 3385 1059 209 1168 2661 4101 1815 4983 1908 1944 2540
