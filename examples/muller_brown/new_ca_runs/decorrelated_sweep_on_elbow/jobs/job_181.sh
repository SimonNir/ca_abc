#!/bin/bash
#SBATCH --job-name=abc_181
#SBATCH --output=logs/abc_181.out
#SBATCH --error=logs/abc_181.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 905 3435 2006 1495 2919 4408 3635 3516 3822 5044 729 3484 4953 3196 4905 4703 1199 1454 1007 710 2002 1220 3833 2445 413 2907
