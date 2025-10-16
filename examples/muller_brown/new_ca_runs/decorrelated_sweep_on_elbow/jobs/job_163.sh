#!/bin/bash
#SBATCH --job-name=abc_163
#SBATCH --output=logs/abc_163.out
#SBATCH --error=logs/abc_163.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2183 1813 1934 4075 4225 1252 1778 1457 3922 660 3551 1500 1891 4594 2285 403 329 3767 438 2506 1883 2487 4090 1574 2082 2784
