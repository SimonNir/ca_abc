#!/bin/bash
#SBATCH --job-name=abc_73
#SBATCH --output=logs/abc_73.out
#SBATCH --error=logs/abc_73.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 1543 1245 1251 556 657 1611 1348 1571 1895 49 162 1890 1110 910 1730 618 1535 1729 401 608
