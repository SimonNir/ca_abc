#!/bin/bash
#SBATCH --job-name=abc_87
#SBATCH --output=logs/abc_87.out
#SBATCH --error=logs/abc_87.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 309 1275 351 1797 527 342 18 1037 1039 1899 1634 1160 357 1060 1852 278 1184 1780 1699 1020
