#!/bin/bash
#SBATCH --job-name=abc_13
#SBATCH --output=logs/abc_13.out
#SBATCH --error=logs/abc_13.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1650 1663 711 347 205 1055 1215 1670 536 1532 1383 1392 196 1618 912 1514 312 212 1962 838
