#!/bin/bash
#SBATCH --job-name=abc_93
#SBATCH --output=logs/abc_93.out
#SBATCH --error=logs/abc_93.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1788 815 1194 339 1693 622 1997 770 872 1881 1394 1633 177 1269 981 1553 1442 784 1291 1804
