#!/bin/bash
#SBATCH --job-name=abc_85
#SBATCH --output=logs/abc_85.out
#SBATCH --error=logs/abc_85.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 286 982 676 454 734 1132 1027 528 11 576 490 710 976
