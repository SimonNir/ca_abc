#!/bin/bash
#SBATCH --job-name=abc_124
#SBATCH --output=logs/abc_124.out
#SBATCH --error=logs/abc_124.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 509 1324 816 633 4360 5069 826 2546 1442 2197 661 1465 2811 340 4298 1641 1616 946 4583 1983 2887 3975 390 3309 3445 4630
