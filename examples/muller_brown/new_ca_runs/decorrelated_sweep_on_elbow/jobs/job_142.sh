#!/bin/bash
#SBATCH --job-name=abc_142
#SBATCH --output=logs/abc_142.out
#SBATCH --error=logs/abc_142.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1327 716 1305 1245 42 2321 4459 1779 3754 4461 3728 2858 4111 452 1262 1028 4547 3712 2252 2134 464 933 1249 3351 3990 2410
