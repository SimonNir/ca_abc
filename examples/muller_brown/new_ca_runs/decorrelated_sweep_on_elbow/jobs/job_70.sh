#!/bin/bash
#SBATCH --job-name=abc_70
#SBATCH --output=logs/abc_70.out
#SBATCH --error=logs/abc_70.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2816 1553 2534 156 2202 1116 964 4572 3606 2793 931 4857 4330 1649 112 899 2032 1599 1858 3636 2973 2908 1673 807 4817 2315
