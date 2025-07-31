#!/bin/bash
#SBATCH --job-name=abc_56
#SBATCH --output=logs/abc_56.out
#SBATCH --error=logs/abc_56.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3246 4942 981 4740 1359 1048 641 1267 3713 2517 3019 117 2653 1304 1083 3500 2574 455 1835 1864 1960 4083 280 4903 1611 2291
