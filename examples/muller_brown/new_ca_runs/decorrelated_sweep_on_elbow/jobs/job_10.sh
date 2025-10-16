#!/bin/bash
#SBATCH --job-name=abc_10
#SBATCH --output=logs/abc_10.out
#SBATCH --error=logs/abc_10.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1736 1868 4765 4095 2876 128 721 4045 3585 3685 2108 3361 1740 4257 303 2423 328 655 927 5097 3470 244 788 527 444 4588
