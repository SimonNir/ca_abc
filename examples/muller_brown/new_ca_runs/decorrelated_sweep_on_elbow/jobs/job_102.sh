#!/bin/bash
#SBATCH --job-name=abc_102
#SBATCH --output=logs/abc_102.out
#SBATCH --error=logs/abc_102.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 919 534 2918 561 2650 1193 4342 1030 4853 4137 2429 4708 2171 3108 4613 1292 3597 3599 2580 4303 3720 5018 1394 895 2444 5050
