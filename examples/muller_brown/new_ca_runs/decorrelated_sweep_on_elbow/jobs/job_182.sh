#!/bin/bash
#SBATCH --job-name=abc_182
#SBATCH --output=logs/abc_182.out
#SBATCH --error=logs/abc_182.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4916 515 2767 2548 3319 3505 3299 1711 1145 3046 2482 4957 3979 1228 4529 3146 963 2952 4879 4855 2789 2902 3879 4262 4969 4560
