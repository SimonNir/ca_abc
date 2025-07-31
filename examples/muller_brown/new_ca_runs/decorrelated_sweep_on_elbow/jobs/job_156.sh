#!/bin/bash
#SBATCH --job-name=abc_156
#SBATCH --output=logs/abc_156.out
#SBATCH --error=logs/abc_156.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4997 260 78 3644 5046 189 1703 2702 962 59 279 2175 3044 912 4452 2063 2356 307 3778 2643 3697 2764 4300 3060 1741 969
