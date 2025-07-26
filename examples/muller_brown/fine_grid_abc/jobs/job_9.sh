#!/bin/bash
#SBATCH --job-name=abc_9
#SBATCH --output=logs/abc_9.out
#SBATCH --error=logs/abc_9.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1558 1966 627 740 881 280 1307 424 1872 1145 1567 599 962 1968 1813 641 1772 1372 338 1963
