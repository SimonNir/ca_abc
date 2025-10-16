#!/bin/bash
#SBATCH --job-name=abc_45
#SBATCH --output=logs/abc_45.out
#SBATCH --error=logs/abc_45.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2442 2372 308 2284 538 166 3150 2480 2103 3640 3544 3439 766 1032 1587 765 507 4890 2820 1542 4752 4140 4065 1042 268 456
