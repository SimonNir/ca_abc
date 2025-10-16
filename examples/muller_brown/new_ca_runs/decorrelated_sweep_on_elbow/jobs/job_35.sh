#!/bin/bash
#SBATCH --job-name=abc_35
#SBATCH --output=logs/abc_35.out
#SBATCH --error=logs/abc_35.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4107 3876 1566 4736 3983 3526 3863 2397 1071 4709 4414 470 4491 3933 4293 1921 1807 219 3446 1627 4306 3178 3282 2899 2932 2303
