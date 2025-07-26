#!/bin/bash
#SBATCH --job-name=abc_41
#SBATCH --output=logs/abc_41.out
#SBATCH --error=logs/abc_41.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 244 240 755 268 1099 1580 1477 645 1735 1632 1220 219 712 1859 271 855 320 107 514 19
