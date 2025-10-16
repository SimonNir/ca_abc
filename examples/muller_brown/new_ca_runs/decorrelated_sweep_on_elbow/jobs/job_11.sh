#!/bin/bash
#SBATCH --job-name=abc_11
#SBATCH --output=logs/abc_11.out
#SBATCH --error=logs/abc_11.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 903 1301 276 73 4014 4021 174 786 3101 2715 4908 1732 1418 3260 757 1306 4365 4245 2115 3434 1231 2459 5092 5021 4820 3678
