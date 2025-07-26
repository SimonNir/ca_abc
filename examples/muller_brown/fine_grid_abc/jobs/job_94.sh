#!/bin/bash
#SBATCH --job-name=abc_94
#SBATCH --output=logs/abc_94.out
#SBATCH --error=logs/abc_94.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 116 156 708 1646 742 243 1898 1530 1811 1216 564 1952 1873 1161 1448 1308 1123 1292 1725 1641
