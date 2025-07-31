#!/bin/bash
#SBATCH --job-name=abc_96
#SBATCH --output=logs/abc_96.out
#SBATCH --error=logs/abc_96.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1115 4012 3824 1582 4273 3496 1692 4471 4152 4863 4862 4672 1834 2771 913 2338 3969 1165 4126 3365 2305 2719 4856 1607 2113 3648
