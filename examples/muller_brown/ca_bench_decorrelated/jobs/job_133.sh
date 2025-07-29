#!/bin/bash
#SBATCH --job-name=abc_133
#SBATCH --output=logs/abc_133.out
#SBATCH --error=logs/abc_133.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 871 784 2487 2412 2493 18 520 1147 1556 595 1926 655 848
