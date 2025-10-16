#!/bin/bash
#SBATCH --job-name=abc_104
#SBATCH --output=logs/abc_104.out
#SBATCH --error=logs/abc_104.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4462 2556 2638 4057 818 4053 1281 95 1250 2738 1975 4446 3826 39 860 2153 2449 2841 3237 1515 827 116 3453 79 4242 4807
