#!/bin/bash
#SBATCH --job-name=abc_82
#SBATCH --output=logs/abc_82.out
#SBATCH --error=logs/abc_82.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 1062 1822 3208 3993 817 1103 2941 3907 2864 2100 113 3925 473 2368 2737 2469 3053 3414 4321 1255 2489 2828 3686 4349 16 4580
