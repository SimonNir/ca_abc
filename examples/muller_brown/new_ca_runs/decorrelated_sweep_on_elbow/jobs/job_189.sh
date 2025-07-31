#!/bin/bash
#SBATCH --job-name=abc_189
#SBATCH --output=logs/abc_189.out
#SBATCH --error=logs/abc_189.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4819 2591 1509 3674 4622 4308 1899 2198 122 1215 4285 4885 4552 1706 1114 1033 3129 17 4927 2604 2928 3377 3614 1871 5060 2606
