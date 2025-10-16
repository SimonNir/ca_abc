#!/bin/bash
#SBATCH --job-name=abc_166
#SBATCH --output=logs/abc_166.out
#SBATCH --error=logs/abc_166.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2341 2551 2957 3201 1372 419 1949 1366 3156 683 4327 2863 549 4302 2670 2673 1022 2266 682 2558 840 1816 4689 1412 3503 2822
