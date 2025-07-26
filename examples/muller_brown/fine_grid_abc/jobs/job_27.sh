#!/bin/bash
#SBATCH --job-name=abc_27
#SBATCH --output=logs/abc_27.out
#SBATCH --error=logs/abc_27.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1211 1923 1373 250 399 321 1186 1724 1472 185 506 211 1775 1117 1614 1919 307 33 1798 944
