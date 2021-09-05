#!/bin/bash
#SBATCH --job-name=c5c4c3c2        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=10G                 # total memory per node (4G per cpu-core is default)
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=hl8967@princeton.edu

HIDDEN=6
LAYERS=2
KERNEL_SIZE=5

module purge
module load anaconda3/2020.7
conda activate qfusion

python -u quantum_learn.py --channels_spatial 'c16' 'c8' 'c4' 'c2' --subsampling 10 --channels_temporal 'c2' 'c5' 'c4' 'c3' 'c2' --kernel_temporal 4 > outputs/smallTCN_c5c4c3c2.out