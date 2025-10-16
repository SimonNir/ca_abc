#!/bin/bash
#SBATCH --job-name=abc_51
#SBATCH --output=logs/abc_51.out
#SBATCH --error=logs/abc_51.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 346 2319 4851 1648 2735 3667 2298 3230 4241 1093 446 876 2675 4368 2362 1335 4456 5015 1402 2501 2154 217 2374 951 1721 3676
