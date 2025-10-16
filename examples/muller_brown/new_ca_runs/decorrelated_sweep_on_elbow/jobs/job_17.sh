#!/bin/bash
#SBATCH --job-name=abc_17
#SBATCH --output=logs/abc_17.out
#SBATCH --error=logs/abc_17.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 708 1234 3752 3694 5062 1256 3316 4013 2970 234 4486 2875 2357 4228 3546 2168 1917 735 2194 622 1150 2727 2798 109 1212 3397
