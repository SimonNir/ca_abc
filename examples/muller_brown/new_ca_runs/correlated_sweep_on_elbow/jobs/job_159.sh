#!/bin/bash
#SBATCH --job-name=abc_159
#SBATCH --output=logs/abc_159.out
#SBATCH --error=logs/abc_159.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1443 124 866 1188 189 1403 829 1399 905
