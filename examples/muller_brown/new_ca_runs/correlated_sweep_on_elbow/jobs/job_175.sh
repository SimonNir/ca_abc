#!/bin/bash
#SBATCH --job-name=abc_175
#SBATCH --output=logs/abc_175.out
#SBATCH --error=logs/abc_175.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1344 460 566 1186 687 432 455 346 875
