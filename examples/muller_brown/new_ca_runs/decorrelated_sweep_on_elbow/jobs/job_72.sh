#!/bin/bash
#SBATCH --job-name=abc_72
#SBATCH --output=logs/abc_72.out
#SBATCH --error=logs/abc_72.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2329 4633 2143 4112 3735 1690 3430 4394 4060 1374 1037 4926 4100 1461 5038 1941 3904 2686 3049 1214 49 2877 1066 3231 1909 4810
