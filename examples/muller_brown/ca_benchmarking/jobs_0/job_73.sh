#!/bin/bash
#SBATCH --job-name=abc_73
#SBATCH --output=logs/abc_73.out
#SBATCH --error=logs/abc_73.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 530 534 420 738 796 772 697 175 618 151 88 628 1104
