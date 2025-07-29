#!/bin/bash
#SBATCH --job-name=abc_170
#SBATCH --output=logs/abc_170.out
#SBATCH --error=logs/abc_170.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1306 430 1466 139 766 1394 324 1179 512
