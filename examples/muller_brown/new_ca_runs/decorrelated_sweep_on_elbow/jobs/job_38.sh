#!/bin/bash
#SBATCH --job-name=abc_38
#SBATCH --output=logs/abc_38.out
#SBATCH --error=logs/abc_38.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 5011 4387 930 488 1926 1584 1365 1121 2130 5029 2809 4806 3865 1629 1930 195 1998 1619 3790 2762 2896 1487 958 2140 4787 3908
