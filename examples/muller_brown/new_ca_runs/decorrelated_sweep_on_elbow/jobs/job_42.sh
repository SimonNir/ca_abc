#!/bin/bash
#SBATCH --job-name=abc_42
#SBATCH --output=logs/abc_42.out
#SBATCH --error=logs/abc_42.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4154 2492 2689 2371 2369 91 4530 2069 4717 2685 1138 3005 3166 3573 4557 991 855 341 4203 2395 1133 5025 4151 567 4618 3218
