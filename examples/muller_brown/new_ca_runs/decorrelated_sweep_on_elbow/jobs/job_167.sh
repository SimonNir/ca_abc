#!/bin/bash
#SBATCH --job-name=abc_167
#SBATCH --output=logs/abc_167.out
#SBATCH --error=logs/abc_167.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3128 3656 2028 4644 99 953 3106 3431 1568 1092 5034 3426 3997 3872 1424 3873 2078 2428 1788 1346 1992 2131 1326 4050 2815 4223
