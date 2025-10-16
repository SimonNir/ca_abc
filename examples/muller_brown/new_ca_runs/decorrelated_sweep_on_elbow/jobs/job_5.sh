#!/bin/bash
#SBATCH --job-name=abc_5
#SBATCH --output=logs/abc_5.out
#SBATCH --error=logs/abc_5.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 514 551 886 4553 4507 5094 3660 3765 45 4161 2060 3286 1187 3478 2549 4264 2968 4034 1623 3149 3852 746 3558 2679 1317 1379
