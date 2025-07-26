#!/bin/bash
#SBATCH --job-name=abc_43
#SBATCH --output=logs/abc_43.out
#SBATCH --error=logs/abc_43.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 963 886 930 1430 1301 459 1942 262 223 1267 791 203 194 1974 301 363 605 87 463 635
