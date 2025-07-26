#!/bin/bash
#SBATCH --job-name=abc_74
#SBATCH --output=logs/abc_74.out
#SBATCH --error=logs/abc_74.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1163 274 259 105 1090 875 234 1207 820 1616 328 738 1816 1144 1562 1625 289 1446 1074 697
