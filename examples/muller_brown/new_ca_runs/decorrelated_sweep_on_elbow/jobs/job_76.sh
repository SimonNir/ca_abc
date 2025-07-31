#!/bin/bash
#SBATCH --job-name=abc_76
#SBATCH --output=logs/abc_76.out
#SBATCH --error=logs/abc_76.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1157 1522 4328 4934 1759 2331 1995 4077 4277 2296 1191 2755 3568 2394 2688 4963 1938 2363 391 4780 4406 3871 1065 435 571 1529
