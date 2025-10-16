#!/bin/bash
#SBATCH --job-name=abc_59
#SBATCH --output=logs/abc_59.out
#SBATCH --error=logs/abc_59.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2959 417 3868 2176 1817 3747 616 798 758 5036 4312 4838 336 310 5024 2565 12 4375 2210 4592 5110 1954 4653 2084 4307 1523
