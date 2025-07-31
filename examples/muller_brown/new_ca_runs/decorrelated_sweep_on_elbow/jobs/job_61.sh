#!/bin/bash
#SBATCH --job-name=abc_61
#SBATCH --output=logs/abc_61.out
#SBATCH --error=logs/abc_61.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3405 896 3184 2729 1419 608 4766 1840 2158 1180 1956 4078 4098 4169 2511 467 2380 2568 1154 4748 2667 4747 61 3197 243 3097
