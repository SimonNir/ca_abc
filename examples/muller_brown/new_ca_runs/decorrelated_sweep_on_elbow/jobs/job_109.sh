#!/bin/bash
#SBATCH --job-name=abc_109
#SBATCH --output=logs/abc_109.out
#SBATCH --error=logs/abc_109.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1486 3810 4194 4894 3222 4304 4119 3823 249 885 4208 3238 2388 4261 4071 1195 4106 1889 582 1330 2020 2560 3937 1997 2000 2111
