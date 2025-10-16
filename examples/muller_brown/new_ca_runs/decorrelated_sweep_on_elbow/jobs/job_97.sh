#!/bin/bash
#SBATCH --job-name=abc_97
#SBATCH --output=logs/abc_97.out
#SBATCH --error=logs/abc_97.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4636 4484 5063 1354 3595 653 1785 4052 1242 2978 3869 3073 2382 3287 3468 3084 4680 3437 2376 4104 3935 560 4678 4030 2885 4795
