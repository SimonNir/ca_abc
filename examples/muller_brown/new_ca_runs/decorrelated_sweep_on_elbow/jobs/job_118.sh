#!/bin/bash
#SBATCH --job-name=abc_118
#SBATCH --output=logs/abc_118.out
#SBATCH --error=logs/abc_118.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4603 197 1333 3967 2852 4268 270 4070 4849 3137 3271 3340 2145 2837 793 3670 3241 288 1756 1075 3559 3080 733 3637 1067 2610
