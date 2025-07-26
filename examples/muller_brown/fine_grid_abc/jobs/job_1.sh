#!/bin/bash
#SBATCH --job-name=abc_1
#SBATCH --output=logs/abc_1.out
#SBATCH --error=logs/abc_1.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1079 1484 1480 562 1426 1720 1414 135 637 1657 1278 491 40 1201 482 384 473 315 1659 589
