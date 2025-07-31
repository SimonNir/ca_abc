#!/bin/bash
#SBATCH --job-name=abc_174
#SBATCH --output=logs/abc_174.out
#SBATCH --error=logs/abc_174.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4733 5016 897 315 1309 2324 3249 264 2414 4734 319 882 2080 4634 824 2205 2035 987 3529 3406 3565 142 4297 2072 4481 3934
