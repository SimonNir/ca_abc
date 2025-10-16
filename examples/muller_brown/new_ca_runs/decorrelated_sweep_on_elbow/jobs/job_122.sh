#!/bin/bash
#SBATCH --job-name=abc_122
#SBATCH --output=logs/abc_122.out
#SBATCH --error=logs/abc_122.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1520 4202 2990 530 4044 699 1263 4534 1733 4143 2206 3147 146 3745 2106 4531 3938 2342 1352 1725 253 2339 3774 769 2792 2019
