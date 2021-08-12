#!/bin/bash
#SBATCH --job-name=tfpennylaneconv1outchannel-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4G per cpu-core is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2020.7
conda activate qfusion-tf

python QConv1Dtf.py