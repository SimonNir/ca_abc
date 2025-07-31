#!/bin/bash
#SBATCH --job-name=abc_91
#SBATCH --output=logs/abc_91.out
#SBATCH --error=logs/abc_91.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1340 2612 4217 2587 992 1742 3228 833 5056 559 4168 2200 4500 1654 1723 3654 873 2057 1990 150 3301 3704 2015 2121 3719 639
