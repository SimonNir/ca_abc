#!/bin/bash
#SBATCH --job-name=abc_104
#SBATCH --output=logs/abc_104.out
#SBATCH --error=logs/abc_104.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 785 270 1209 605 1463 487 924 425 439
