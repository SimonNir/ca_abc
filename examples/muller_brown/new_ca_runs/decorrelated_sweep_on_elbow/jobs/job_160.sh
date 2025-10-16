#!/bin/bash
#SBATCH --job-name=abc_160
#SBATCH --output=logs/abc_160.out
#SBATCH --error=logs/abc_160.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 711 3522 24 2304 4313 1660 3755 578 843 2888 5080 175 1989 3057 478 3618 821 2939 5085 294 4499 3887 680 4116 4741 1685
