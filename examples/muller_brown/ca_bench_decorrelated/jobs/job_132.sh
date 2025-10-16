#!/bin/bash
#SBATCH --job-name=abc_132
#SBATCH --output=logs/abc_132.out
#SBATCH --error=logs/abc_132.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1498 1101 1880 77 712 970 881 1678 1544 1965 1878 910 2441
