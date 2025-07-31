#!/bin/bash
#SBATCH --job-name=abc_13
#SBATCH --output=logs/abc_13.out
#SBATCH --error=logs/abc_13.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4231 5102 2853 3772 4598 1638 4410 4434 3182 3987 29 4830 1558 2302 60 1172 4816 4097 1717 3569 945 4679 4068 1169 3641 1441
