#!/bin/bash
#SBATCH --job-name=abc_162
#SBATCH --output=logs/abc_162.out
#SBATCH --error=logs/abc_162.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 57 1143 2336 3525 355 4775 5033 2750 4213 1730 4123 937 1964 1923 187 1530 3579 125 2092 2146 4390 1760 1036 2463 791 1745
