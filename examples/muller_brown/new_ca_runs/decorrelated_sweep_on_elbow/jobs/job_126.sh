#!/bin/bash
#SBATCH --job-name=abc_126
#SBATCH --output=logs/abc_126.out
#SBATCH --error=logs/abc_126.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 915 5083 4860 2161 1241 2023 3330 3950 3710 4528 857 149 3532 2141 4259 693 360 3120 3158 2619 3078 2977 2972 1555 220 2223
