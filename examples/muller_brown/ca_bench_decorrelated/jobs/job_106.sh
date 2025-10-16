#!/bin/bash
#SBATCH --job-name=abc_106
#SBATCH --output=logs/abc_106.out
#SBATCH --error=logs/abc_106.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1255 2422 1008 2476 274 1257
