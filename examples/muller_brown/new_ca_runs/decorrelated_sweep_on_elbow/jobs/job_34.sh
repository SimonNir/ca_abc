#!/bin/bash
#SBATCH --job-name=abc_34
#SBATCH --output=logs/abc_34.out
#SBATCH --error=logs/abc_34.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3663 3759 388 2289 726 1541 2824 1953 3317 3314 3680 192 1965 4244 4211 1664 47 2412 1401 995 2870 3800 4987 3165 3970 3645
