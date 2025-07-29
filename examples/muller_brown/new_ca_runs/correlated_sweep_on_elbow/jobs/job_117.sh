#!/bin/bash
#SBATCH --job-name=abc_117
#SBATCH --output=logs/abc_117.out
#SBATCH --error=logs/abc_117.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 916 651 403 265 669 1325 436 1389 1546
