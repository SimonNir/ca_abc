#!/bin/bash
#SBATCH --job-name=abc_24
#SBATCH --output=logs/abc_24.out
#SBATCH --error=logs/abc_24.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1550 2966 1102 1537 4932 628 1482 819 3548 275 3333 1838 584 2434 4278 977 1708 1634 1688 1337 3464 2163 3274 4976 4669 3460
