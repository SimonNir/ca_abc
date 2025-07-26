#!/bin/bash
#SBATCH --job-name=abc_47
#SBATCH --output=logs/abc_47.out
#SBATCH --error=logs/abc_47.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 584 1327 481 97 1289 1741 1031 681 1722 988 983 1061 802 1542 8 1883 612 233 1565 16
