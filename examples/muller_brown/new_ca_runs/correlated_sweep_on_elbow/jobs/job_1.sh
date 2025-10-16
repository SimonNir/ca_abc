#!/bin/bash
#SBATCH --job-name=abc_1
#SBATCH --output=logs/abc_1.out
#SBATCH --error=logs/abc_1.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1128 410 1232 142 559 428 175 628 964
