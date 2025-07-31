#!/bin/bash
#SBATCH --job-name=abc_62
#SBATCH --output=logs/abc_62.out
#SBATCH --error=logs/abc_62.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4985 4474 4909 134 1499 4467 2600 1753 3914 1011 1576 4889 5084 589 3944 1179 3052 3918 1579 4384 168 1869 1040 3181 588 1258
