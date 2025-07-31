#!/bin/bash
#SBATCH --job-name=abc_176
#SBATCH --output=logs/abc_176.out
#SBATCH --error=logs/abc_176.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3248 678 1563 2843 1254 2555 4427 4627 923 3041 1438 1089 2803 2683 858 1687 107 3355 1970 2981 2471 254 2120 173 4092 3605
