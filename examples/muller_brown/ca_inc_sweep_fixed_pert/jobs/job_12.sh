#!/bin/bash
#SBATCH --job-name=abc_12
#SBATCH --output=logs/abc_12.out
#SBATCH --error=logs/abc_12.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 785 808 330 239 187 428 215 755 675
