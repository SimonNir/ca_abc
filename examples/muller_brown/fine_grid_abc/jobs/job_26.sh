#!/bin/bash
#SBATCH --job-name=abc_26
#SBATCH --output=logs/abc_26.out
#SBATCH --error=logs/abc_26.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1252 1413 273 861 1779 38 531 679 1583 168 1441 1310 1258 709 128 1338 1406 545 1364 1469
