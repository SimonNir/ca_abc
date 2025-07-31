#!/bin/bash
#SBATCH --job-name=abc_14
#SBATCH --output=logs/abc_14.out
#SBATCH --error=logs/abc_14.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4400 1898 1260 1455 2717 1409 1825 1135 627 891 4756 2101 3561 2571 1064 5030 3615 1620 3768 3589 3379 1319 3268 4655 3506 2177
