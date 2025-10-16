#!/bin/bash
#SBATCH --job-name=abc_20
#SBATCH --output=logs/abc_20.out
#SBATCH --error=logs/abc_20.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1382 4147 316 4385 2668 1027 3267 2308 80 3114 5100 3984 1716 1047 3110 3580 4460 893 640 4522 1095 797 1632 970 1058 292
