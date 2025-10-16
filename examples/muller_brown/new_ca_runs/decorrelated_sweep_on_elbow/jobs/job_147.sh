#!/bin/bash
#SBATCH --job-name=abc_147
#SBATCH --output=logs/abc_147.out
#SBATCH --error=logs/abc_147.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1407 997 181 1077 3507 4659 167 3909 1139 2333 1096 3306 2230 2008 72 2306 2709 2347 542 3031 317 4029 3422 2274 4238 2700
