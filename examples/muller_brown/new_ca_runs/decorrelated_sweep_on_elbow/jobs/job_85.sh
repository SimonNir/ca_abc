#!/bin/bash
#SBATCH --job-name=abc_85
#SBATCH --output=logs/abc_85.out
#SBATCH --error=logs/abc_85.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4099 624 2614 2162 3067 1720 2881 2493 1223 940 1665 3266 4754 4172 1081 1353 1483 304 3994 1935 2191 3841 1860 2573 5012 1416
