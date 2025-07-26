#!/bin/bash
#SBATCH --job-name=abc_11
#SBATCH --output=logs/abc_11.out
#SBATCH --error=logs/abc_11.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 683 1475 221 1470 1660 428 319 866 414 1388 877 1499 1325 544 188 228 1044 1705 502 893
