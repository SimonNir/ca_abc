#!/bin/bash
#SBATCH --job-name=abc_195
#SBATCH --output=logs/abc_195.out
#SBATCH --error=logs/abc_195.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3699 1430 3931 2722 311 2821 412 2231 947 465 332 4234 2330 1432 2317 3847 183 4355 512 4409 239 2128 3639 1222 4108 2173
