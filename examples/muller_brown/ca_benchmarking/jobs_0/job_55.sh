#!/bin/bash
#SBATCH --job-name=abc_55
#SBATCH --output=logs/abc_55.out
#SBATCH --error=logs/abc_55.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 116 463 347 681 303 691 1063 1160 271 326 874 154 829
