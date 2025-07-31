#!/bin/bash
#SBATCH --job-name=abc_136
#SBATCH --output=logs/abc_136.out
#SBATCH --error=logs/abc_136.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4413 3702 2422 842 2818 1994 4865 3016 4716 3262 2589 4121 3594 834 756 2465 1489 3692 4336 599 1761 379 2758 314 1967 182
