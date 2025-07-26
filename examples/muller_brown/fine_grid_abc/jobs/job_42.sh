#!/bin/bash
#SBATCH --job-name=abc_42
#SBATCH --output=logs/abc_42.out
#SBATCH --error=logs/abc_42.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 108 1071 389 198 756 1604 497 508 1681 682 1624 326 1991 1910 692 1831 37 1205 903 1612
