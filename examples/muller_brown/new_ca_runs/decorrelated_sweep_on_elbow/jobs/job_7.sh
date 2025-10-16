#!/bin/bash
#SBATCH --job-name=abc_7
#SBATCH --output=logs/abc_7.out
#SBATCH --error=logs/abc_7.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1002 1035 1743 3926 990 4162 1608 277 2267 4710 2053 2576 1765 1845 4981 2906 3350 3076 1658 4320 4570 1229 4549 448 3117 4489
