#!/bin/bash
#SBATCH --job-name=abc_15
#SBATCH --output=logs/abc_15.out
#SBATCH --error=logs/abc_15.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1659 5040 2265 2766 4999 3083 1202 4118 5057 2488 1447 1820 3221 2579 1164 2849 5052 1511 1300 1880 547 4265 2909 1610 4639 2886
