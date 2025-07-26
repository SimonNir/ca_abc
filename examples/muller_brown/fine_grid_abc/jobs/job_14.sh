#!/bin/bash
#SBATCH --job-name=abc_14
#SBATCH --output=logs/abc_14.out
#SBATCH --error=logs/abc_14.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 673 236 642 167 607 184 1591 1864 1187 798 1142 1787 1629 1764 1509 968 1204 297 1050 746
