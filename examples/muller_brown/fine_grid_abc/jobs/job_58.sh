#!/bin/bash
#SBATCH --job-name=abc_58
#SBATCH --output=logs/abc_58.out
#SBATCH --error=logs/abc_58.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 392 1978 1889 1850 856 900 1999 1051 1177 1533 494 688 48 654 400 1589 1036 1402 1666 1943
