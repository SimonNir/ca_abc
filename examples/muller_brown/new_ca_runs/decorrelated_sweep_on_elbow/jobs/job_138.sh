#!/bin/bash
#SBATCH --job-name=abc_138
#SBATCH --output=logs/abc_138.out
#SBATCH --error=logs/abc_138.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4441 1039 521 3281 3177 4936 1339 659 2468 5096 3951 4220 439 1929 4842 1737 4700 4624 3845 4315 4227 1971 384 1247 4353 4574
