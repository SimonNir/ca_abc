#!/bin/bash
#SBATCH --job-name=abc_53
#SBATCH --output=logs/abc_53.out
#SBATCH --error=logs/abc_53.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1490 1062 260 1539 630 1229 619 1055 496
