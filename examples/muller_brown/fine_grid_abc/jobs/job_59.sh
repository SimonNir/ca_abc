#!/bin/bash
#SBATCH --job-name=abc_59
#SBATCH --output=logs/abc_59.out
#SBATCH --error=logs/abc_59.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 989 1501 1038 1135 1478 680 112 1257 1065 1370 466 648 1596 1382 30 1203 1770 3 447 1807
