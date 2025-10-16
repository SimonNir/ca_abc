#!/bin/bash
#SBATCH --job-name=abc_191
#SBATCH --output=logs/abc_191.out
#SBATCH --error=logs/abc_191.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1573 2833 4762 943 1564 4314 77 2457 3232 5007 3621 914 2500 365 4192 1186 2585 380 778 808 2182 1903 1591 1581 2181 1118
