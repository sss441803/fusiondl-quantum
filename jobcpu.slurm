#!/bin/bash
#SBATCH --job-name=qfusion-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G                 # total memory per node (4G per cpu-core is default)
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=hl8967@princeton.edu

module purge
module load anaconda3/2020.7
conda activate qfusion

python -u quantum_learn.py > quantum_learn.out