#!/bin/bash
#SBATCH --job-name=abc_23
#SBATCH --output=logs/abc_23.out
#SBATCH --error=logs/abc_23.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1671 193 4444 3396 3378 4785 3331 839 3783 928 309 1355 245 3943 4696 2699 4237 796 1846 851 2385 581 2724 2807 2299 4232
