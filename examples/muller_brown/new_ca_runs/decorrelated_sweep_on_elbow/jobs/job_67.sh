#!/bin/bash
#SBATCH --job-name=abc_67
#SBATCH --output=logs/abc_67.out
#SBATCH --error=logs/abc_67.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 845 3598 579 416 1108 2261 3485 250 461 4974 4402 4626 2041 4745 323 4821 2286 4031 3514 1291 1109 2910 1814 4125 1427 4955
