#!/bin/bash
#SBATCH --job-name=abc_32
#SBATCH --output=logs/abc_32.out
#SBATCH --error=logs/abc_32.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1606 1615 420 584 765 444 1203 310 849
