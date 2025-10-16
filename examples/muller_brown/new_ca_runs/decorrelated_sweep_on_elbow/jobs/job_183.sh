#!/bin/bash
#SBATCH --job-name=abc_183
#SBATCH --output=logs/abc_183.out
#SBATCH --error=logs/abc_183.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1367 1429 1119 64 2588 3332 3995 1417 2741 4777 4003 2520 1812 1185 1194 2393 3564 3469 986 1313 2030 3508 3633 2220 2209 1358
