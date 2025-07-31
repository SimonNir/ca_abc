#!/bin/bash
#SBATCH --job-name=abc_64
#SBATCH --output=logs/abc_64.out
#SBATCH --error=logs/abc_64.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4088 5065 3359 2710 1636 3875 137 1652 4892 1151 2328 3627 1793 596 4724 2956 203 1024 4145 321 4513 848 1128 3899 2845 3134
