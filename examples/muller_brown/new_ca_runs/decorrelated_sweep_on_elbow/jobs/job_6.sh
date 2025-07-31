#!/bin/bash
#SBATCH --job-name=abc_6
#SBATCH --output=logs/abc_6.out
#SBATCH --error=logs/abc_6.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1056 3706 5086 4281 3520 5116 2730 1349 3753 1918 3212 476 3199 663 2227 196 2545 3725 3109 2484 3924 600 68 3880 1747 1336
