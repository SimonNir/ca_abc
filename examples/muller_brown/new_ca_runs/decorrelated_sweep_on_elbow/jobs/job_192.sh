#!/bin/bash
#SBATCH --job-name=abc_192
#SBATCH --output=logs/abc_192.out
#SBATCH --error=logs/abc_192.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4980 3846 2712 4144 4743 427 3829 4416 575 2927 3930 3042 3107 4876 525 597 1693 3658 2005 1809 4511 656 2625 2437 2402 3374
