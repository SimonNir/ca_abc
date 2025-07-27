#!/bin/bash
#SBATCH --job-name=abc_41
#SBATCH --output=logs/abc_41.out
#SBATCH --error=logs/abc_41.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1049 139 932 740 913 78 133 362 494 1119 311 473 85
