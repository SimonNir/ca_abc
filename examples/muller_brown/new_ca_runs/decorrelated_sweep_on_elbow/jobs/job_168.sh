#!/bin/bash
#SBATCH --job-name=abc_168
#SBATCH --output=logs/abc_168.out
#SBATCH --error=logs/abc_168.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2391 1288 1862 4282 3771 4243 93 1549 284 4973 1505 2483 4790 4761 2690 1320 4915 4555 1141 4791 2701 1211 3375 333 1216 1020
