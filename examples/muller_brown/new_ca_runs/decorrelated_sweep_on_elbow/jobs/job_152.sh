#!/bin/bash
#SBATCH --job-name=abc_152
#SBATCH --output=logs/abc_152.out
#SBATCH --error=logs/abc_152.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1700 750 3708 615 689 3001 4910 3818 2613 2931 849 120 852 4727 665 4538 2189 5005 437 2384 741 3325 2929 1079 5009 3345
