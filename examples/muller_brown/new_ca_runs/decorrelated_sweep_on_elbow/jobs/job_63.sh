#!/bin/bash
#SBATCH --job-name=abc_63
#SBATCH --output=logs/abc_63.out
#SBATCH --error=logs/abc_63.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2112 1159 2507 647 1278 1746 3088 2705 3690 1920 363 770 110 2408 3285 480 88 5 3538 3684 3171 2706 841 4605 3751 2498
