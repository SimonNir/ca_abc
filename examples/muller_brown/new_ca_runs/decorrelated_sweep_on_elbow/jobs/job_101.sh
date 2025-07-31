#!/bin/bash
#SBATCH --job-name=abc_101
#SBATCH --output=logs/abc_101.out
#SBATCH --error=logs/abc_101.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 506 3812 566 755 1322 3578 918 4253 1562 1669 3717 5109 318 1209 982 3992 5098 4577 1719 1833 121 381 4781 1435 4455 523
