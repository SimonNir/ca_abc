#!/bin/bash
#SBATCH --job-name=abc_55
#SBATCH --output=logs/abc_55.out
#SBATCH --error=logs/abc_55.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 479 317 337 883 1619 1785 792 131 985 165 1042 1318 146 73 50 365 1209 1502 1638 960
