#!/bin/bash
#SBATCH --job-name=abc_114
#SBATCH --output=logs/abc_114.out
#SBATCH --error=logs/abc_114.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3135 577 2728 4958 3399 1318 1493 3779 4808 1950 490 3124 4426 2065 787 4470 3220 3724 1210 1556 5104 2857 3764 722 3198 4150
