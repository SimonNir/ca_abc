#!/bin/bash
#SBATCH --job-name=abc_78
#SBATCH --output=logs/abc_78.out
#SBATCH --error=logs/abc_78.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 670 470 845 1202 94 765 950 715 549 943 1970 477 1056 1369 32 984 398 450 1685 53
