#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 4

module load anaconda3
conda activate torch
# Traverse
export OMPI_MCA_btl="tcp,self,vader"
module load cudatoolkit
module load cudnn/cuda-10.1/7.6.1   # cudann---> module not found (ANSWER: use "cudnn")
module load openmpi/gcc/3.1.4/64
module load hdf5/gcc/openmpi-3.1.4/1.10.5
srun python torch_learn.py
