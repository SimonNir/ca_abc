#!/bin/bash
#SBATCH --job-name=abc_90
#SBATCH --output=logs/abc_90.out
#SBATCH --error=logs/abc_90.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1303 1617 2017 4493 2407 4383 1129 2370 2787 645 1879 1902 2093 326 2283 672 4504 4067 4802 694 4797 3179 3486 4288 2711 1175
