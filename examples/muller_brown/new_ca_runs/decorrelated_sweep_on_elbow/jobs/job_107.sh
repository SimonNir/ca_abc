#!/bin/bash
#SBATCH --job-name=abc_107
#SBATCH --output=logs/abc_107.out
#SBATCH --error=logs/abc_107.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3311 2595 481 1773 1462 2823 4880 1873 330 1881 4858 649 954 2598 2530 2051 451 4401 2911 4673 3570 98 5051 4051 752 4404
