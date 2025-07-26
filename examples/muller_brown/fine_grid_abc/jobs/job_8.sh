#!/bin/bash
#SBATCH --job-name=abc_8
#SBATCH --output=logs/abc_8.out
#SBATCH --error=logs/abc_8.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1757 1444 1507 47 1067 102 603 515 799 1085 1916 551 1534 1606 1520 1958 1546 451 439 1794
