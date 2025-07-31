#!/bin/bash
#SBATCH --job-name=abc_113
#SBATCH --output=logs/abc_113.out
#SBATCH --error=logs/abc_113.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3510 2318 290 1086 2043 4251 3515 4609 679 3008 2617 4344 1697 950 2894 4019 1295 836 4952 3683 1595 2126 3864 242 3501 863
