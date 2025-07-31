#!/bin/bash
#SBATCH --job-name=abc_173
#SBATCH --output=logs/abc_173.out
#SBATCH --error=logs/abc_173.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1681 1588 4338 2014 4396 40 2535 4770 2677 4715 3204 2013 2129 424 4033 2734 3099 3169 4179 218 4483 1981 366 1961 2949 2117
