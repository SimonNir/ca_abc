#!/bin/bash
#SBATCH --job-name=abc_31
#SBATCH --output=logs/abc_31.out
#SBATCH --error=logs/abc_31.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1422 2392 3665 411 3582 462 484 4291 3608 210 4087 4811 667 505 3945 5072 1569 4069 2219 3890 4649 1605 696 241 2859 4371
