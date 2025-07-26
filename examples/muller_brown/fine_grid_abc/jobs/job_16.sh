#!/bin/bash
#SBATCH --job-name=abc_16
#SBATCH --output=logs/abc_16.out
#SBATCH --error=logs/abc_16.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1259 296 1415 98 890 310 46 1537 1976 769 1948 810 93 1994 525 1425 1401 1487 794 1603
