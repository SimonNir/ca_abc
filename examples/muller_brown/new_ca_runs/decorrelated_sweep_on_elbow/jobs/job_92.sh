#!/bin/bash
#SBATCH --job-name=abc_92
#SBATCH --output=logs/abc_92.out
#SBATCH --error=logs/abc_92.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2950 212 3590 996 1865 1763 3256 3877 2485 5077 2087 11 2930 2521 3392 2310 587 879 1308 3441 1521 1174 907 1091 1906 4159
