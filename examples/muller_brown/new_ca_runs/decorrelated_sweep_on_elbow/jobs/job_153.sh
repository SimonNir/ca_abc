#!/bin/bash
#SBATCH --job-name=abc_153
#SBATCH --output=logs/abc_153.out
#SBATCH --error=logs/abc_153.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1466 4301 2095 603 4395 1380 3998 201 358 540 2377 2221 2123 2416 5032 2253 2753 5117 1639 3357 777 866 3622 4615 706 1714
