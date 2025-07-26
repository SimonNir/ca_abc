#!/bin/bash
#SBATCH --job-name=abc_23
#SBATCH --output=logs/abc_23.out
#SBATCH --error=logs/abc_23.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 501 104 530 1949 1287 1153 83 916 1516 1323 1778 1897 659 1014 330 1077 20 113 1305 1887
