#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=MP-Edit
#SBATCH --gres=gpu:3
#SBATCH --constraint=gpu_12gb
#SBATCH --mail-type=END
#SBATCH --mail-user=johnnyma@nyu.edu
#SBATCH --output=slurm_MP-Edit%j.out
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

SRCDIR=$/misc/vlgscratch4/HeGroup/jlm10003

# Activate the conda environment
source ~/.bashrc
conda activate mpe

# Execute the script
# Usage: python3.8 MP-Edit.py <dataset> <number of lines> <which action to take>

python3.8 /misc/vlgscratch4/HeGroup/jlm10003/MP-Edit.py IMDB 141 Predict
 

# And we're done!
