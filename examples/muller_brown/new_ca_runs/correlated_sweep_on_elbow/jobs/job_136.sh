#!/bin/bash
#SBATCH --job-name=abc_136
#SBATCH --output=logs/abc_136.out
#SBATCH --error=logs/abc_136.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 40 815 1107 53 1088 26 523 326 633
