#!/bin/bash
#SBATCH --job-name=abc_79
#SBATCH --output=logs/abc_79.out
#SBATCH --error=logs/abc_79.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4055 2994 1072 2743 4786 3671 1289 94 1273 337 1913 1176 1152 1383 2432 576 1332 2050 732 3363 3471 2697 3244 2971 4451 1772
