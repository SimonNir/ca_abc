#!/bin/bash
#SBATCH --job-name=abc_172
#SBATCH --output=logs/abc_172.out
#SBATCH --error=logs/abc_172.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1293 3481 4263 3429 670 3175 1991 935 62 3334 1055 1821 4687 1751 4586 3502 27 483 2829 949 4437 1405 4026 4214 3587 3401
