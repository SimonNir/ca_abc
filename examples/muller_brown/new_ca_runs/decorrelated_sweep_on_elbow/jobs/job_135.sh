#!/bin/bash
#SBATCH --job-name=abc_135
#SBATCH --output=logs/abc_135.out
#SBATCH --error=logs/abc_135.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4684 856 2674 3343 2736 939 2721 3227 3488 1544 4 1049 4004 3347 1655 772 4891 3214 1571 2403 457 4509 3035 3170 3376 3091
