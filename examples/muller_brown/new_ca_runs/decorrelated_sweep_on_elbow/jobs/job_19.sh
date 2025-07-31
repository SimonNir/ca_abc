#!/bin/bash
#SBATCH --job-name=abc_19
#SBATCH --output=logs/abc_19.out
#SBATCH --error=logs/abc_19.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 614 2664 978 1755 1762 867 537 1626 4063 5105 2411 4671 687 2834 3213 4642 436 1911 4046 1689 2835 1943 375 1201 734 3252
