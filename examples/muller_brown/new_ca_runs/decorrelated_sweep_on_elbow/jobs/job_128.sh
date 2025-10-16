#!/bin/bash
#SBATCH --job-name=abc_128
#SBATCH --output=logs/abc_128.out
#SBATCH --error=logs/abc_128.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 783 662 1624 1661 2801 4675 1342 3673 19 4280 2462 4494 4813 3242 2871 2327 541 135 2682 4419 1806 4949 3304 4453 1937 2277
