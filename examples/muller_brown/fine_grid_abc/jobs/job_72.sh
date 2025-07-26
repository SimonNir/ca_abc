#!/bin/bash
#SBATCH --job-name=abc_72
#SBATCH --output=logs/abc_72.out
#SBATCH --error=logs/abc_72.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 498 842 144 257 172 1568 272 490 593 1169 1907 1824 344 1857 614 1452 1981 997 734 325
