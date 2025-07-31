#!/bin/bash
#SBATCH --job-name=abc_52
#SBATCH --output=logs/abc_52.out
#SBATCH --error=logs/abc_52.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1423 902 1310 5058 1996 2562 1484 2742 3022 434 3575 4008 4558 224 1277 764 1378 4875 3677 3535 1969 4898 3457 4378 4660 1400
