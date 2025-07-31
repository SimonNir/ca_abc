#!/bin/bash
#SBATCH --job-name=abc_86
#SBATCH --output=logs/abc_86.out
#SBATCH --error=logs/abc_86.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4826 2409 1012 272 1519 373 2732 1131 4982 4989 474 4954 4079 2866 4345 3303 1274 4546 3207 3731 4482 4874 3264 4062 4468 4105
