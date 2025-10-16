#!/bin/bash
#SBATCH --job-name=abc_21
#SBATCH --output=logs/abc_21.out
#SBATCH --error=logs/abc_21.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2794 820 2569 4640 1915 975 188 104 1585 3624 1264 5026 1951 2922 3577 3744 1248 4007 3836 3327 1323 2435 2064 2073 2104 3058
