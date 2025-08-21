#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:00:30
#SBATCH --mem=500m

#SBATCH -p burst

#SBATCH -A birthright

#SBATCH --job-name=test

#SBATCH --mail-user=nirenbergsd@ornl.gov
#SBATCH --mail-type=END

#SBATCH -o test.out

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "test complete"