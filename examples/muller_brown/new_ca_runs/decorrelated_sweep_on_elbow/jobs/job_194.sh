#!/bin/bash
#SBATCH --job-name=abc_194
#SBATCH --output=logs/abc_194.out
#SBATCH --error=logs/abc_194.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2586 4629 4472 3733 1572 3917 3688 1613 1280 1240 459 34 494 4764 4283 1296 853 338 3741 4492 2636 2772 368 1474 4479 810
