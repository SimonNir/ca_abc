#!/bin/bash
#SBATCH --job-name=abc_159
#SBATCH --output=logs/abc_159.out
#SBATCH --error=logs/abc_159.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4757 92 3029 4184 482 989 3278 650 3804 1829 4469 2652 535 3216 4645 941 1841 2708 1200 2991 418 123 3721 3977 4110 3257
