#!/bin/bash
#SBATCH --job-name=abc_106
#SBATCH --output=logs/abc_106.out
#SBATCH --error=logs/abc_106.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1478 612 2254 394 4994 2912 4438 1021 4730 2999 2047 423 917 1978 1345 4714 3254 4230 2856 2242 3968 3596 3193 3380 4612 674
