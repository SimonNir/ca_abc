#!/bin/bash
#SBATCH --job-name=abc_32
#SBATCH --output=logs/abc_32.out
#SBATCH --error=logs/abc_32.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 390 982 676 1073 771 1214 932 1698 1122 1823 78 1615 1087 487 1422 468 276 1784 1986 729
