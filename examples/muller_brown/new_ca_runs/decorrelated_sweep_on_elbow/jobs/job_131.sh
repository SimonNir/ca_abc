#!/bin/bash
#SBATCH --job-name=abc_131
#SBATCH --output=logs/abc_131.out
#SBATCH --error=logs/abc_131.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1381 3322 124 4017 2761 2270 2503 3815 2326 707 1329 1777 2883 3360 409 1155 226 3115 5089 1307 2137 1266 1826 1452 4913 3632
