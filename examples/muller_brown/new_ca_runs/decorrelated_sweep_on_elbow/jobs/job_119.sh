#!/bin/bash
#SBATCH --job-name=abc_119
#SBATCH --output=logs/abc_119.out
#SBATCH --error=logs/abc_119.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 5071 2777 297 2526 2164 4221 4527 3999 754 2044 3825 4565 4219 3920 3972 4988 2455 4731 1163 3562 3737 1000 2282 2671 1294 1361
