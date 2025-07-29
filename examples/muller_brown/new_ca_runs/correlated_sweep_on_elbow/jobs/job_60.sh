#!/bin/bash
#SBATCH --job-name=abc_60
#SBATCH --output=logs/abc_60.out
#SBATCH --error=logs/abc_60.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1396 828 881 1175 288 79 89 593 1173
