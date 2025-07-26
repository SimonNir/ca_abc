#!/bin/bash
#SBATCH --job-name=abc_7
#SBATCH --output=logs/abc_7.out
#SBATCH --error=logs/abc_7.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1835 1197 1389 1691 1192 748 370 12 1217 1512 965 404 256 1447 993 809 471 1221 1157 1955
