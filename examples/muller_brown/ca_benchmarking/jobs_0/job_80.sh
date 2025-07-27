#!/bin/bash
#SBATCH --job-name=abc_80
#SBATCH --output=logs/abc_80.out
#SBATCH --error=logs/abc_80.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 315 657 426 132 711 468 666 402 137 285 446 455 389
