#!/bin/bash
#SBATCH --job-name=abc_65
#SBATCH --output=logs/abc_65.out
#SBATCH --error=logs/abc_65.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1671 1449 1920 937 1759 1844 785 1678 1561 54 1510 281 507 1321 452 231 1743 1868 1769 835
