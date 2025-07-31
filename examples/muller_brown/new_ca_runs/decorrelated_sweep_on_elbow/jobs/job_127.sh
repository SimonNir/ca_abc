#!/bin/bash
#SBATCH --job-name=abc_127
#SBATCH --output=logs/abc_127.out
#SBATCH --error=logs/abc_127.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3335 90 652 4977 1227 4173 3476 3259 723 3757 4965 2192 5088 1502 3705 3630 1149 2061 3087 1144 2703 2593 2453 4501 1644 3240
