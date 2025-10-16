#!/bin/bash
#SBATCH --job-name=abc_66
#SBATCH --output=logs/abc_66.out
#SBATCH --error=logs/abc_66.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3782 4049 4091 4763 2239 3588 2714 610 1184 4854 3144 872 3018 1839 4872 1795 371 4114 3239 4628 4182 522 574 4319 2360 21
