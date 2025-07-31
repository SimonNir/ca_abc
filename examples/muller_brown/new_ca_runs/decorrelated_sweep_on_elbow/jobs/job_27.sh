#!/bin/bash
#SBATCH --job-name=abc_27
#SBATCH --output=logs/abc_27.out
#SBATCH --error=logs/abc_27.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4674 3966 1456 1415 442 4508 4218 4772 2251 1125 1539 3436 1376 3620 965 700 795 1734 1496 4212 4690 440 2800 5013 932 1769
