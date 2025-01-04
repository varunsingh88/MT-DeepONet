#!/bin/bash
#SBATCH --job-name=Fisher_DeepONet_exp2
#SBATCH -t 02:00:00
#SBATCH --mem=64g
##SBATCH -p 3090-gcondo
##SBATCH -p a6000-gcondo
#SBATCH -p h100-gcondo
#SBATCH --account=gk-h100-gcondo
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

#export SINGULARITY_BINDPATH="/gpfs/scratch,/gpfs/data"
export APPTAINER_BINDPATH="/oscar/scratch,/oscar/data"

#singularity exec --nv /users/vkumar24/data/vkumar24/container_imgs/tf2_9.simg python -u main_PINN.py
singularity exec --nv /users/vkumar24/data/vkumar24/container_imgs/tf2_14.simg python -u main.py