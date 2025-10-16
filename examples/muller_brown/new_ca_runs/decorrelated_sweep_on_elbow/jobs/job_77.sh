#!/bin/bash
#SBATCH --job-name=abc_77
#SBATCH --output=logs/abc_77.out
#SBATCH --error=logs/abc_77.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2985 2003 607 1111 1054 376 2512 2349 4518 690 3272 1299 3748 2309 548 3613 3689 2354 3002 4391 3957 4584 4478 4498 8 2196
