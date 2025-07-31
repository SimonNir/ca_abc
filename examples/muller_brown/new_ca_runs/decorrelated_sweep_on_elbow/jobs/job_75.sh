#!/bin/bash
#SBATCH --job-name=abc_75
#SBATCH --output=logs/abc_75.out
#SBATCH --error=logs/abc_75.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3296 111 724 926 2622 4284 3552 4789 4914 1625 3113 686 4788 4648 968 2916 4233 200 737 3787 4380 2618 33 4367 331 2071
