#!/bin/bash
#SBATCH --job-name=abc_155
#SBATCH --output=logs/abc_155.out
#SBATCH --error=logs/abc_155.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3482 4918 3897 1705 306 3448 814 3781 592 3312 830 815 1962 524 4376 1450 3811 4907 2170 1397 3 2805 287 1668 1874 3809
