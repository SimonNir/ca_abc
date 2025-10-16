#!/bin/bash
#SBATCH --job-name=abc_32
#SBATCH --output=logs/abc_32.out
#SBATCH --error=logs/abc_32.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 281 5066 2243 4931 4002 3284 3492 4526 806 4986 3288 1464 479 1161 671 2257 3475 1986 2992 4440 4837 4267 106 4834 5003 2774
