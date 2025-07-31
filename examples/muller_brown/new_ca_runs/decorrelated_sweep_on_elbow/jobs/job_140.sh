#!/bin/bash
#SBATCH --job-name=abc_140
#SBATCH --output=logs/abc_140.out
#SBATCH --error=logs/abc_140.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 466 4246 4962 151 2208 2190 1373 1802 590 4399 3224 4760 3275 5119 2184 3121 1890 3093 2452 4495 2443 2090 4540 2746 1534 4311
