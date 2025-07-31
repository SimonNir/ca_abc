#!/bin/bash
#SBATCH --job-name=abc_141
#SBATCH --output=logs/abc_141.out
#SBATCH --error=logs/abc_141.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2802 4080 1577 799 3032 4272 1666 3631 1171 751 1531 2681 503 4551 2739 430 2831 2543 4742 994 2052 2980 296 1311 170 1985
