#!/bin/bash
#SBATCH --job-name=abc_184
#SBATCH --output=logs/abc_184.out
#SBATCH --error=logs/abc_184.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3466 2038 3867 1554 3229 4477 1726 3000 3668 4463 1590 1510 1236 1433 1882 4668 2379 100 3015 3936 392 4442 2745 3127 2974 2366
