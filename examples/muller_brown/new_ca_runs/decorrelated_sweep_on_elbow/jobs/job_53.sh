#!/bin/bash
#SBATCH --job-name=abc_53
#SBATCH --output=logs/abc_53.out
#SBATCH --error=logs/abc_53.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3794 4667 3253 1506 763 3419 3410 837 2401 3739 235 1123 775 739 3928 2415 4036 3055 3454 749 5070 4431 3777 4859 825 3851
