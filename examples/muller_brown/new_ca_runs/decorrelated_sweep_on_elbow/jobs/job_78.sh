#!/bin/bash
#SBATCH --job-name=abc_78
#SBATCH --output=logs/abc_78.out
#SBATCH --error=logs/abc_78.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4369 1384 516 3962 85 2293 3646 2596 4536 4545 1046 3738 4964 269 1350 4109 785 3638 3451 4823 4625 2951 4643 1508 1390 2222
