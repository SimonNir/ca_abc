#!/bin/bash
#SBATCH --job-name=abc_82
#SBATCH --output=logs/abc_82.out
#SBATCH --error=logs/abc_82.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py 610 467 852 227 1113 976 1025 632 1630 65 1175 1236 1847 1796 1314 920 1550 1586 825 282
