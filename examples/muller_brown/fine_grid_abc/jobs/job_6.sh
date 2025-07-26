#!/bin/bash
#SBATCH --job-name=abc_6
#SBATCH --output=logs/abc_6.out
#SBATCH --error=logs/abc_6.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1654 726 393 1939 1738 860 267 1941 334 1009 1557 733 1517 133 819 1820 1455 665 1855 1378
