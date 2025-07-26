#!/bin/bash
#SBATCH --job-name=abc_44
#SBATCH --output=logs/abc_44.out
#SBATCH --error=logs/abc_44.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 563 1335 360 1133 1867 1445 446 1602 1354 1652 772 1527 478 827 413 1711 704 1435 160 362
