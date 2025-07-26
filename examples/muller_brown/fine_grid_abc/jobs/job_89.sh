#!/bin/bash
#SBATCH --job-name=abc_89
#SBATCH --output=logs/abc_89.out
#SBATCH --error=logs/abc_89.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1000 969 768 964 862 1465 290 1944 75 1904 1656 1360 275 978 225 1874 1200 812 1708 1132
